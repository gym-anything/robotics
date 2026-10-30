"""Topology-aware flat-lay reset seeds for sewn garments.

A flat garment is not a stack of coincident two-dimensional pattern pieces.
The body of a T-shirt is a closed shell, each sleeve is a tube with an open
cuff, and the neck rib is a second sewn loop.  This module authors that
three-dimensional reset geometry before the material metric preconditioner and
the physical solver run.

Strategies are registered per construction.  Unsupported garments fail
explicitly instead of falling back to a rectangle or a generic overlapping
panel atlas.  The authored layer separation is a reset-only numerical
preconditioner, not a calibrated claim about textile thickness.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from .garment_assembly import RigidPanelSeedPose, SeedAssembly
from .garment_asset import GarmentConstruction
from .garment_meshing import GarmentPatternMesh


@dataclass(frozen=True)
class FlatLayResetConfig:
    """Numerical reset geometry kept separate from physical material data."""

    body_layer_separation_m: float = 8.0e-3
    neck_opening_center_offset_fraction: float = 0.5

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.body_layer_separation_m)
            or self.body_layer_separation_m <= 0.0
        ):
            raise ValueError("Flat-lay body separation must be finite and positive")
        if (
            not math.isfinite(self.neck_opening_center_offset_fraction)
            or self.neck_opening_center_offset_fraction <= 0.0
        ):
            raise ValueError(
                "Flat-lay neck-centre offset fraction must be finite and positive"
            )


@dataclass(frozen=True)
class FlatLaySeedReport:
    construction_id: str
    strategy_id: str
    reset_only_numerical_preconditioner: bool
    body_layer_separation_m: float
    constrained_vertex_count_by_panel: Mapping[str, int]
    harmonic_vertex_count_by_panel: Mapping[str, int]
    inter_panel_seam_rms_gap_m: float
    maximum_inter_panel_seam_rms_gap_m: float
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "construction_id": self.construction_id,
            "strategy_id": self.strategy_id,
            "reset_only_numerical_preconditioner": (
                self.reset_only_numerical_preconditioner
            ),
            "body_layer_separation_m": self.body_layer_separation_m,
            "constrained_vertex_count_by_panel": dict(
                self.constrained_vertex_count_by_panel
            ),
            "harmonic_vertex_count_by_panel": dict(
                self.harmonic_vertex_count_by_panel
            ),
            "inter_panel_seam_rms_gap_m": self.inter_panel_seam_rms_gap_m,
            "maximum_inter_panel_seam_rms_gap_m": (
                self.maximum_inter_panel_seam_rms_gap_m
            ),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class FlatLaySeedResult:
    seed: SeedAssembly
    report: FlatLaySeedReport


def _panel_local_indices(
    pattern_mesh: GarmentPatternMesh,
    panel_id: str,
    global_indices: np.ndarray,
) -> np.ndarray:
    start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
    indices = np.asarray(global_indices, dtype=np.int32)
    if indices.size and (
        int(np.min(indices)) < start or int(np.max(indices)) >= stop
    ):
        raise ValueError(f"A constraint leaves panel {panel_id!r}")
    return indices - start


def _harmonic_panel_extension(
    panel_triangles: np.ndarray,
    vertex_count: int,
    constrained_targets: Mapping[int, np.ndarray],
    *,
    base_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Uniform graph-harmonic extension of landmark displacements.

    A nonzero base preserves an undeformed planar cut piece away from the
    constrained construction paths.  Extending absolute coordinates with a
    combinatorial Laplacian can fold an irregular triangulation even when all
    boundary targets are valid; extending displacements avoids that artifact.
    """

    if not constrained_targets:
        raise ValueError("A harmonic flat-lay panel needs Dirichlet constraints")
    constrained = np.asarray(sorted(constrained_targets), dtype=np.int32)
    if int(np.min(constrained)) < 0 or int(np.max(constrained)) >= vertex_count:
        raise ValueError("A harmonic constraint references a missing panel vertex")
    targets = np.asarray(
        [constrained_targets[int(index)] for index in constrained],
        dtype=np.float64,
    )
    if targets.shape != (len(constrained), 3) or not np.all(np.isfinite(targets)):
        raise ValueError("Harmonic flat-lay targets must be finite 3-D points")

    if base_positions is None:
        base = np.zeros((vertex_count, 3), dtype=np.float64)
    else:
        base = np.asarray(base_positions, dtype=np.float64)
        if base.shape != (vertex_count, 3) or not np.all(np.isfinite(base)):
            raise ValueError("Harmonic base positions must be finite 3-D points")
    constrained_displacements = targets - base[constrained]
    result = base.copy()
    result[constrained] = targets
    free = np.setdiff1d(
        np.arange(vertex_count, dtype=np.int32),
        constrained,
        assume_unique=True,
    )
    if len(free) == 0:
        return result

    try:
        from scipy.sparse import coo_matrix  # noqa: PLC0415
        from scipy.sparse.linalg import spsolve  # noqa: PLC0415
    except ImportError as exception:  # pragma: no cover - bootstrap dependency
        raise RuntimeError(
            "Topology-aware flat-lay seeds require SciPy; run "
            "laundry/bootstrap_newton.ps1 first."
        ) from exception

    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
    for a_raw, b_raw, c_raw in np.asarray(panel_triangles, dtype=np.int32):
        a, b, c = int(a_raw), int(b_raw), int(c_raw)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    if any(not neighbors[int(vertex)] for vertex in free):
        raise ValueError("A free panel vertex is disconnected from the cloth mesh")

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for vertex, adjacent in enumerate(neighbors):
        rows.append(vertex)
        columns.append(vertex)
        values.append(float(len(adjacent)))
        for neighbor in adjacent:
            rows.append(vertex)
            columns.append(neighbor)
            values.append(-1.0)
    laplacian = coo_matrix(
        (values, (rows, columns)),
        shape=(vertex_count, vertex_count),
        dtype=np.float64,
    ).tocsr()
    system = laplacian[free][:, free]
    right_hand_side = -(
        laplacian[free][:, constrained] @ constrained_displacements
    )
    solved = spsolve(system, right_hand_side)
    solved = np.asarray(solved, dtype=np.float64)
    if solved.shape != (len(free), 3) or not np.all(np.isfinite(solved)):
        raise RuntimeError("Harmonic flat-lay extension failed")
    result[free] += solved
    return result


def _mean_targets(
    candidates: Mapping[int, list[np.ndarray]],
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for vertex, values in candidates.items():
        points = np.asarray(values, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Flat-lay constraint candidates must be 3-D")
        # Multiple construction paths legitimately meet at a cut corner.  A
        # front/back cap crown contributes equal and opposite layer heights;
        # averaging those targets authors the developable sleeve fold there.
        result[int(vertex)] = np.mean(points, axis=0)
    return result


def _seam_gap_metrics(
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
    positions: np.ndarray,
) -> tuple[float, float]:
    gaps: list[float] = []
    for seam, stitch in zip(construction.seams, pattern_mesh.stitches):
        if seam.id != stitch.seam_id:
            raise ValueError("Construction seams and compiled stitches are misaligned")
        if seam.side_a.panel_id == seam.side_b.panel_id:
            continue
        delta = (
            positions[stitch.vertex_pairs[:, 0]]
            - positions[stitch.vertex_pairs[:, 1]]
        )
        gaps.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return (
        float(np.sqrt(np.mean(np.square(gaps)))) if gaps else 0.0,
        max(gaps, default=0.0),
    )


def _panel_mesh(
    pattern_mesh: GarmentPatternMesh,
    panel_id: str,
):
    return pattern_mesh.panel_meshes[pattern_mesh.panel_ids.index(panel_id)]


def _interpolate_panel_surface(
    pattern_mesh: GarmentPatternMesh,
    positions_m: np.ndarray,
    panel_id: str,
    material_points_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Barycentrically sample an authored panel and its world normals."""

    mesh = _panel_mesh(pattern_mesh, panel_id)
    start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
    world = positions_m[start:stop]
    queries = np.asarray(material_points_m, dtype=np.float64)
    sampled = np.empty((len(queries), 3), dtype=np.float64)
    normals = np.empty_like(sampled)
    material_triangles = mesh.material_coordinates_m[mesh.triangles]
    vertex_normals = np.zeros_like(world)
    for triangle in mesh.triangles:
        world_triangle = world[triangle]
        face_normal = np.cross(
            world_triangle[1] - world_triangle[0],
            world_triangle[2] - world_triangle[0],
        )
        for vertex in triangle:
            vertex_normals[int(vertex)] += face_normal
    vertex_normal_length = np.linalg.norm(vertex_normals, axis=1)
    if np.any(vertex_normal_length <= 1.0e-12):
        raise ValueError(f"Host panel {panel_id!r} has an undefined vertex normal")
    vertex_normals /= vertex_normal_length[:, None]
    for query_index, query in enumerate(queries):
        selected: tuple[np.ndarray, np.ndarray] | None = None
        best_minimum_weight = -math.inf
        inside = False
        for triangle_index, (a, b, c) in enumerate(material_triangles):
            matrix = np.column_stack((b - a, c - a))
            determinant = float(np.linalg.det(matrix))
            if abs(determinant) <= 1.0e-14:
                continue
            uv = np.linalg.solve(matrix, query - a)
            weights = np.asarray((1.0 - uv[0] - uv[1], uv[0], uv[1]))
            minimum_weight = float(np.min(weights))
            if minimum_weight >= -1.0e-9:
                selected = (mesh.triangles[triangle_index], weights)
                inside = True
                break
            if minimum_weight > best_minimum_weight:
                best_minimum_weight = minimum_weight
                selected = (mesh.triangles[triangle_index], weights)
        if selected is None or (not inside and best_minimum_weight < -5.0e-3):
            # ``best_minimum_weight`` remains -inf for a true miss; the loose
            # fallback only covers floating-point queries on an outline edge.
            raise ValueError(
                f"A material query leaves host panel {panel_id!r}: {query.tolist()}"
            )
        triangle, weights = selected
        world_triangle = world[triangle]
        sampled[query_index] = weights @ world_triangle
        normal = weights @ vertex_normals[triangle]
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-12:
            raise ValueError(f"Host panel {panel_id!r} has a degenerate triangle")
        normals[query_index] = normal / norm
    return sampled, normals


def _constraints_from_host_seams(
    pattern_mesh: GarmentPatternMesh,
    positions_m: np.ndarray,
    panel_id: str,
    seam_ids: tuple[str, ...],
) -> dict[int, np.ndarray]:
    """Copy stitch-corresponding targets from already-authored host panels."""

    start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
    stitch_by_id = {stitch.seam_id: stitch for stitch in pattern_mesh.stitches}
    candidates: dict[int, list[np.ndarray]] = {}
    for seam_id in seam_ids:
        stitch = stitch_by_id[seam_id]
        for first_raw, second_raw in stitch.vertex_pairs:
            first, second = int(first_raw), int(second_raw)
            if start <= first < stop and not (start <= second < stop):
                local, host = first - start, second
            elif start <= second < stop and not (start <= first < stop):
                local, host = second - start, first
            else:
                raise ValueError(
                    f"Host seam {seam_id!r} does not connect panel {panel_id!r} "
                    "to exactly one authored panel"
                )
            target = positions_m[host]
            if not np.all(np.isfinite(target)):
                raise ValueError(
                    f"Host seam {seam_id!r} reads an unauthored panel vertex"
                )
            candidates.setdefault(local, []).append(target.copy())
    return _mean_targets(candidates)


def _affine_panel_base(
    material_coordinates_m: np.ndarray,
    constrained_targets: Mapping[int, np.ndarray],
    *,
    z_offset_m: float = 0.0,
) -> np.ndarray:
    """Fit a stable planar base before harmonic seam displacement."""

    constrained = np.asarray(sorted(constrained_targets), dtype=np.int32)
    source = material_coordinates_m[constrained]
    target = np.asarray(
        [constrained_targets[int(index)] for index in constrained],
        dtype=np.float64,
    )
    design = np.column_stack((source, np.ones(len(source))))
    if len(source) >= 3 and np.linalg.matrix_rank(design) == 3:
        coefficients, *_ = np.linalg.lstsq(design, target[:, :2], rcond=None)
        xy = np.column_stack(
            (material_coordinates_m, np.ones(len(material_coordinates_m)))
        ) @ coefficients
    else:
        # A tie anchor or one straight seam does not constrain a full affine
        # map.  Align its longest chord and retain material scale elsewhere.
        distances = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
        first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
        source_delta = source[second] - source[first]
        target_delta = target[second, :2] - target[first, :2]
        source_length = float(np.linalg.norm(source_delta))
        target_length = float(np.linalg.norm(target_delta))
        if min(source_length, target_length) <= 1.0e-12:
            raise ValueError("A panel attachment chord is degenerate")
        source_angle = math.atan2(source_delta[1], source_delta[0])
        target_angle = math.atan2(target_delta[1], target_delta[0])
        angle = target_angle - source_angle
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        scale = target_length / source_length
        translation = target[first, :2] - scale * source[first] @ rotation.T
        xy = scale * material_coordinates_m @ rotation.T + translation
    z = np.full(
        len(material_coordinates_m),
        float(np.mean(target[:, 2])) + float(z_offset_m),
        dtype=np.float64,
    )
    return np.column_stack((xy, z))


def _author_attached_panel(
    pattern_mesh: GarmentPatternMesh,
    positions_m: np.ndarray,
    panel_id: str,
    seam_ids: tuple[str, ...],
    *,
    base_positions: np.ndarray | None = None,
    base_z_offset_m: float = 0.0,
) -> tuple[int, int]:
    """Author a panel sewn to an already placed host surface.

    ``base_z_offset_m`` is a layer *lift away from the attachment*, not a
    rigid translation.  A rigid z offset is annihilated by harmonic extension
    whenever every Dirichlet value lies on the host (the constant correction
    is itself harmonic).  That old behaviour silently put pockets, yokes, and
    overlays exactly on top of their hosts.  The normalized material-space
    distance field below is zero at every stitched vertex and reaches the
    requested lift at the most remote vertex, so the shared seam remains
    exact while the two physical cloth layers peel apart.
    """

    start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
    mesh = _panel_mesh(pattern_mesh, panel_id)
    constraints = _constraints_from_host_seams(
        pattern_mesh, positions_m, panel_id, seam_ids
    )
    if base_positions is None:
        base = _affine_panel_base(
            mesh.material_coordinates_m,
            constraints,
            z_offset_m=0.0,
        )
    else:
        base = np.asarray(base_positions, dtype=np.float64)
    authored = _harmonic_panel_extension(
        mesh.triangles,
        mesh.vertex_count,
        constraints,
        base_positions=base,
    )
    if base_z_offset_m != 0.0:
        constrained = np.asarray(sorted(constraints), dtype=np.int32)
        delta = (
            mesh.material_coordinates_m[:, None, :]
            - mesh.material_coordinates_m[constrained][None, :, :]
        )
        distance = np.min(np.linalg.norm(delta, axis=2), axis=1)
        maximum_distance = float(np.max(distance))
        if maximum_distance <= 1.0e-12:
            raise ValueError(
                f"Attached panel {panel_id!r} has no material extent away from its seam"
            )
        fraction = np.clip(distance / maximum_distance, 0.0, 1.0)
        # A lapped or patch seam needs a finite departure angle.  A smoothstep
        # has zero slope at the stitch and therefore leaves the two finite
        # surfaces tangent/coincident for their first element row.  The sine
        # ramp is still smooth at the free region while departing the shared
        # stitch immediately, as a real folded seam allowance does.
        lift = np.sin(0.5 * math.pi * fraction)
        authored[:, 2] += float(base_z_offset_m) * lift
        authored[constrained] = np.asarray(
            [constraints[int(index)] for index in constrained],
            dtype=np.float64,
        )
    positions_m[start:stop] = authored
    return len(constraints), mesh.vertex_count - len(constraints)


def _author_shaped_neck_facing(
    pattern_mesh: GarmentPatternMesh,
    positions_m: np.ndarray,
    panel_id: str,
    *,
    attach_seam_id: str,
    surface_sign: float,
    lift_m: float,
) -> tuple[int, int]:
    """Lay one shaped facing on its body surface and roll it at shoulders.

    The front and back facings occupy opposite sides of the flattened shell.
    Their shoulder seams, however, meet around the cut edge at the common
    mid-plane.  Explicitly authoring that transition avoids the nonphysical
    twisted corner produced by independently lifting two panels and then
    averaging their welded seam particles.
    """

    if surface_sign not in (-1.0, 1.0) or not 0.0 < lift_m:
        raise ValueError("A shaped facing needs a signed surface and positive lift")
    start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
    mesh = _panel_mesh(pattern_mesh, panel_id)
    attach_constraints = _constraints_from_host_seams(
        pattern_mesh, positions_m, panel_id, (attach_seam_id,)
    )
    base = _affine_panel_base(mesh.material_coordinates_m, attach_constraints)
    base = _harmonic_panel_extension(
        mesh.triangles,
        mesh.vertex_count,
        attach_constraints,
        base_positions=base,
    )

    attach_global = pattern_mesh.path_vertices[(panel_id, "neck_attach")]
    attach_local = attach_global - start
    attach_first = attach_constraints[int(attach_local[0])]
    attach_last = attach_constraints[int(attach_local[-1])]

    # The two cut pieces have slightly different endpoint normals.  Their
    # sewn shoulder edge uses the mean of those genuine pattern offsets, so
    # both copies arrive at exactly the same free corner without stretching
    # either one to an arbitrary radial construction.
    shoulder_displacements: dict[str, np.ndarray] = {}
    for path_id in ("left_shoulder_join", "right_shoulder_join"):
        displacements: list[np.ndarray] = []
        for facing_id in ("front_neck_facing", "back_neck_facing"):
            facing_start, _ = pattern_mesh.panel_vertex_ranges[facing_id]
            facing_mesh = _panel_mesh(pattern_mesh, facing_id)
            local = (
                pattern_mesh.path_vertices[(facing_id, path_id)] - facing_start
            )
            coordinates = facing_mesh.material_coordinates_m[local]
            if path_id == "left_shoulder_join":
                displacements.append(coordinates[-1] - coordinates[0])
            else:
                displacements.append(coordinates[0] - coordinates[-1])
        shoulder_displacement = np.mean(displacements, axis=0)
        shoulder_displacements[path_id] = shoulder_displacement

    def rolled_shoulder(attach: np.ndarray, path_id: str) -> np.ndarray:
        target = attach.copy()
        target[:2] += shoulder_displacements[path_id]
        target[2] = 0.0
        return target

    left_free = rolled_shoulder(attach_last, "left_shoulder_join")
    right_free = rolled_shoulder(attach_first, "right_shoulder_join")
    candidates: dict[int, list[np.ndarray]] = {
        index: [target.copy()] for index, target in attach_constraints.items()
    }

    free_global = pattern_mesh.path_vertices[(panel_id, "free_edge")]
    free_fraction = pattern_mesh.path_fractions[(panel_id, "free_edge")]
    for global_vertex, fraction in zip(free_global, free_fraction):
        local = int(global_vertex) - start
        target = base[local].copy()
        target[2] = surface_sign * lift_m * math.sin(math.pi * float(fraction))
        if float(fraction) <= 1.0e-12:
            target = left_free.copy()
        elif float(fraction) >= 1.0 - 1.0e-12:
            target = right_free.copy()
        candidates.setdefault(local, []).append(target)

    for path_id, first_target, second_target in (
        ("left_shoulder_join", attach_last, left_free),
        ("right_shoulder_join", right_free, attach_first),
    ):
        vertices = pattern_mesh.path_vertices[(panel_id, path_id)]
        fractions = pattern_mesh.path_fractions[(panel_id, path_id)]
        for global_vertex, fraction in zip(vertices, fractions):
            local = int(global_vertex) - start
            target = (
                (1.0 - float(fraction)) * first_target
                + float(fraction) * second_target
            )
            candidates.setdefault(local, []).append(target)

    constraints = _mean_targets(candidates)
    positions_m[start:stop] = _harmonic_panel_extension(
        mesh.triangles,
        mesh.vertex_count,
        constraints,
        base_positions=base,
    )
    return len(constraints), mesh.vertex_count - len(constraints)


def _author_flattened_sleeve(
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
    positions_m: np.ndarray,
    sleeve_id: str,
    *,
    side_sign: float,
    cap_seam_ids: tuple[str, str],
    opening_path_id: str,
    half_height: float,
) -> tuple[int, int]:
    """Author an open-ended sewn sleeve as a low-profile elliptical tube.

    Below the sleeve-cap base, material arclength is mapped to ellipse
    arclength with the same circumference.  This is a genuine embedded tube:
    the two underarm cut edges meet at one generator, the opposite generator
    is the explicit fold axis, and front/back cloth never begins coincident.
    Only the cap patch is harmonically blended into the compiled armhole
    correspondences.  The ellipse aspect ratio is reset-only clearance, not a
    claim about a relaxed sleeve's physical thickness.
    """

    start, stop = pattern_mesh.panel_vertex_ranges[sleeve_id]
    mesh = _panel_mesh(pattern_mesh, sleeve_id)
    constraints = _constraints_from_host_seams(
        pattern_mesh, positions_m, sleeve_id, cap_seam_ids
    )
    candidates: dict[int, list[np.ndarray]] = {
        index: [target.copy()] for index, target in constraints.items()
    }

    def add(global_vertex: int, target: np.ndarray) -> None:
        local = int(global_vertex) - start
        if not 0 <= local < stop - start:
            raise ValueError(f"A sleeve target leaves panel {sleeve_id!r}")
        candidates.setdefault(local, []).append(np.asarray(target, dtype=np.float64))

    cap_endpoint_vertices = {
        int(pattern_mesh.path_vertices[(sleeve_id, path_id)][endpoint])
        for path_id in ("cap_front", "cap_back")
        for endpoint in (0, -1)
    }
    for global_vertex in cap_endpoint_vertices:
        local = global_vertex - start
        if local in candidates:
            for target in candidates[local]:
                target[2] = 0.0

    crown_global = int(pattern_mesh.path_vertices[(sleeve_id, "cap_front")][-1])
    armpit_global = int(pattern_mesh.path_vertices[(sleeve_id, "cap_front")][0])
    crown = np.mean(candidates[crown_global - start], axis=0)[:2]
    armpit = np.mean(candidates[armpit_global - start], axis=0)[:2]
    cap_chord = armpit - crown
    cap_chord_length = float(np.linalg.norm(cap_chord))
    panel = construction.panel_map[sleeve_id]
    opening_half_width = 0.5 * panel.path(opening_path_id).length_m
    if opening_half_width <= 0.0:
        raise ValueError(f"Sleeve {sleeve_id!r} opening cannot be degenerate")
    cap_direction = cap_chord / cap_chord_length
    cap_normal = np.asarray((-cap_direction[1], cap_direction[0]))
    width_ratio = opening_half_width / cap_chord_length
    if width_ratio < 0.98:
        opening_normal = (
            -width_ratio * cap_direction
            + math.sqrt(1.0 - width_ratio**2) * side_sign * cap_normal
        )
    else:
        # A bell cuff is intentionally wider than its cap chord.  The
        # isosceles construction used by a straight sleeve has no real square
        # root in that case; lay the flare normal to the cap and let the
        # material-width field carry its genuine widening.
        opening_normal = side_sign * cap_normal
    longitudinal = side_sign * np.asarray(
        (opening_normal[1], -opening_normal[0])
    )
    underarm_length = 0.5 * (
        panel.path("underarm_front").length_m
        + panel.path("underarm_back").length_m
    )
    opening_seam = armpit + underarm_length * longitudinal
    opening_fold = opening_seam + opening_half_width * opening_normal

    if "flat_lay_fold_axis" not in panel.paths:
        raise ValueError(
            f"Sleeve {sleeve_id!r} requires an explicit reset fold axis"
        )
    del half_height  # superseded by the arclength-preserving tube

    cap_base_y = float(panel.path("cap_front").points_m[0][1])
    opening_y = float(np.mean(np.asarray(panel.path(opening_path_id).points_m)[:, 1]))
    if cap_base_y <= opening_y:
        raise ValueError(f"Sleeve {sleeve_id!r} has no positive tube length")

    def boundary_x_at_y(path_id: str, y: float) -> float:
        points = np.asarray(panel.path(path_id).points_m, dtype=np.float64)
        candidates_x: list[float] = []
        for first, second in zip(points[:-1], points[1:]):
            low, high = sorted((float(first[1]), float(second[1])))
            if low - 1.0e-10 <= y <= high + 1.0e-10:
                dy = float(second[1] - first[1])
                if abs(dy) <= 1.0e-12:
                    candidates_x.extend((float(first[0]), float(second[0])))
                else:
                    fraction = (y - float(first[1])) / dy
                    candidates_x.append(
                        float(first[0] + fraction * (second[0] - first[0]))
                    )
        if not candidates_x:
            raise ValueError(
                f"Sleeve boundary {path_id!r} does not cover y={y:.6f} m"
            )
        return float(np.mean(candidates_x))

    # One normalized ellipse table is reused at every longitudinal station.
    # Scaling both semi-axes later makes its perimeter exactly equal to the
    # local cut-panel circumference.
    ellipse_aspect = 0.10
    angle_samples = np.linspace(0.0, 2.0 * math.pi, 4097)
    speed = np.sqrt(
        np.sin(angle_samples) ** 2
        + ellipse_aspect**2 * np.cos(angle_samples) ** 2
    )
    cumulative = np.zeros_like(angle_samples)
    delta_angle = angle_samples[1] - angle_samples[0]
    cumulative[1:] = np.cumsum(
        0.5 * (speed[:-1] + speed[1:]) * delta_angle
    )
    unit_perimeter = float(cumulative[-1])
    cumulative /= unit_perimeter

    local = mesh.material_coordinates_m
    tube_mask = local[:, 1] <= cap_base_y + 1.0e-10
    for local_index in np.flatnonzero(tube_mask):
        x, y = map(float, local[local_index])
        longitudinal_fraction = np.clip(
            (y - opening_y) / (cap_base_y - opening_y), 0.0, 1.0
        )
        back_x = boundary_x_at_y("underarm_back", y)
        front_x = boundary_x_at_y("underarm_front", y)
        circumference = front_x - back_x
        if circumference <= 1.0e-10:
            raise ValueError(f"Sleeve {sleeve_id!r} has a collapsed tube section")
        arclength_fraction = np.clip(
            (x - back_x) / circumference, 0.0, 1.0
        )
        angle = float(
            np.interp(arclength_fraction, cumulative, angle_samples)
        )
        semi_major = circumference / unit_perimeter
        semi_minor = ellipse_aspect * semi_major
        seam_xy = (
            opening_seam
            + longitudinal_fraction * (armpit - opening_seam)
        )
        target = np.asarray(
            (
                *(
                    seam_xy
                    + semi_major * (1.0 - math.cos(angle)) * opening_normal
                ),
                -semi_minor * math.sin(angle),
            ),
            dtype=np.float64,
        )
        add(start + int(local_index), target)

    constraints = _mean_targets(candidates)
    fold_path = panel.path("flat_lay_fold_axis")
    fold_start = np.asarray(fold_path.points_m[0], dtype=np.float64)
    fold_end = np.asarray(fold_path.points_m[-1], dtype=np.float64)
    fold_delta = fold_end - fold_start
    fold_length_squared = float(np.dot(fold_delta, fold_delta))
    if fold_length_squared <= 1.0e-12:
        raise ValueError(f"Sleeve {sleeve_id!r} has a degenerate reset fold axis")
    fraction = ((local - fold_start[None, :]) @ fold_delta) / fold_length_squared
    base = np.zeros((mesh.vertex_count, 3), dtype=np.float64)
    base[:, :2] = (
        opening_fold[None, :]
        + fraction[:, None] * (crown - opening_fold)[None, :]
        - np.abs(local[:, 0, None]) * opening_normal[None, :]
    )
    positions_m[start:stop] = _harmonic_panel_extension(
        mesh.triangles,
        mesh.vertex_count,
        constraints,
        base_positions=base,
    )
    return len(constraints), mesh.vertex_count - len(constraints)


def _author_rectangular_band(
    pattern_mesh: GarmentPatternMesh,
    positions_m: np.ndarray,
    panel_id: str,
    *,
    attach_seam_ids: tuple[str, ...],
    attach_path_id: str,
    far_path_id: str,
    far_offset_vector_m: tuple[float, float, float] | None = None,
    offset_toward_centroid_m: float | None = None,
    far_z_offset_m: float = 0.0,
    attachment_departure_m: float = 0.0,
    attachment_departure_scale_m: float = 8.0e-3,
) -> tuple[int, int]:
    """Lay a sewn cuff, facing, stand, or collar as a ruled strip."""

    if (
        not math.isfinite(attachment_departure_m)
        or attachment_departure_m < 0.0
        or not math.isfinite(attachment_departure_scale_m)
        or attachment_departure_scale_m <= 0.0
    ):
        raise ValueError("Band attachment departure controls are invalid")

    start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
    mesh = _panel_mesh(pattern_mesh, panel_id)
    candidates: dict[int, list[np.ndarray]] = {}
    attach_constraints = _constraints_from_host_seams(
        pattern_mesh, positions_m, panel_id, attach_seam_ids
    )
    for index, target in attach_constraints.items():
        candidates.setdefault(index, []).append(target.copy())

    attach_global = pattern_mesh.path_vertices[(panel_id, attach_path_id)]
    attach_local = attach_global - start
    attach_x = mesh.material_coordinates_m[attach_local, 0]
    attach_targets = np.asarray(
        [attach_constraints[int(index)] for index in attach_local],
        dtype=np.float64,
    )
    order = np.argsort(attach_x)
    attach_x = attach_x[order]
    attach_targets = attach_targets[order]

    def attached_at_x(x: float) -> np.ndarray:
        return np.asarray(
            [np.interp(x, attach_x, attach_targets[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )

    centroid = np.mean(attach_targets[:, :2], axis=0)
    attach_x_min = float(np.min(attach_x))
    attach_x_max = float(np.max(attach_x))
    attach_x_span = attach_x_max - attach_x_min
    if attach_x_span <= 1.0e-12:
        raise ValueError("A ruled band attachment has no longitudinal extent")
    inward_normal_sign = 1.0
    if offset_toward_centroid_m is not None:
        attachment_xy = attach_targets[:, :2]
        attachment_tangent = np.empty_like(attachment_xy)
        attachment_tangent[0] = attachment_xy[1] - attachment_xy[0]
        attachment_tangent[-1] = attachment_xy[-1] - attachment_xy[-2]
        if len(attachment_xy) > 2:
            attachment_tangent[1:-1] = attachment_xy[2:] - attachment_xy[:-2]
        attachment_tangent_length = np.linalg.norm(attachment_tangent, axis=1)
        valid_tangent = attachment_tangent_length > 1.0e-10
        if np.count_nonzero(valid_tangent) < 2:
            raise ValueError("A ruled band attachment has too few valid tangents")
        attachment_tangent[valid_tangent] /= attachment_tangent_length[
            valid_tangent, None
        ]
        candidate_normal = np.column_stack(
            (-attachment_tangent[valid_tangent, 1], attachment_tangent[valid_tangent, 0])
        )
        # Select one orientation for the *whole* curve.  Choosing the side at
        # every sample independently makes the normal flip wherever a concave
        # neckline's centroid vector becomes tangent, producing a zig-zag collar.
        orientation_score = float(
            np.sum(
                candidate_normal
                * (centroid[None, :] - attachment_xy[valid_tangent])
            )
        )
        inward_normal_sign = 1.0 if orientation_score >= 0.0 else -1.0

    def far_at_x(x: float) -> np.ndarray:
        attached = attached_at_x(x)
        if far_offset_vector_m is not None:
            if far_z_offset_m != 0.0:
                raise ValueError(
                    "A ruled band cannot specify z in both far-edge offset modes"
                )
            return attached + np.asarray(far_offset_vector_m, dtype=np.float64)
        if offset_toward_centroid_m is None:
            raise ValueError("A ruled band needs one far-edge offset mode")
        # A band cross-grain is locally normal to its stitched centreline.
        # The globally selected normal sign stays continuous around a concave
        # neck curve; a negative offset intentionally selects the opposite
        # (turn-down/outward) side without pointwise normal flips.
        derivative_half_width = max(1.0e-6, 1.0e-3 * attach_x_span)
        before = max(attach_x_min, float(x) - derivative_half_width)
        after = min(attach_x_max, float(x) + derivative_half_width)
        tangent = attached_at_x(after)[:2] - attached_at_x(before)[:2]
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length <= 1.0e-10:
            raise ValueError("A ruled band attachment has a degenerate tangent")
        tangent /= tangent_length
        direction = inward_normal_sign * np.asarray(
            (-tangent[1], tangent[0])
        )
        result = attached.copy()
        result[:2] += float(offset_toward_centroid_m) * direction
        result[2] += float(far_z_offset_m)
        return result

    attach_y = float(
        np.mean(mesh.material_coordinates_m[attach_global - start, 1])
    )
    far_global = pattern_mesh.path_vertices[(panel_id, far_path_id)]
    far_y = float(np.mean(mesh.material_coordinates_m[far_global - start, 1]))
    height = far_y - attach_y
    if abs(height) <= 1.0e-12:
        raise ValueError("A ruled band attachment and far edge cannot coincide")
    base = np.empty((mesh.vertex_count, 3), dtype=np.float64)
    for index, (x, y) in enumerate(mesh.material_coordinates_m):
        fraction = (float(y) - attach_y) / height
        base[index] = (1.0 - fraction) * attached_at_x(float(x)) + fraction * far_at_x(float(x))

    for path_id in (attach_path_id, far_path_id):
        ids = pattern_mesh.path_vertices[(panel_id, path_id)]
        for global_vertex in ids:
            local_index = int(global_vertex) - start
            x, y = mesh.material_coordinates_m[local_index]
            fraction = (float(y) - attach_y) / height
            target = (1.0 - fraction) * attached_at_x(float(x)) + fraction * far_at_x(float(x))
            candidates.setdefault(local_index, []).append(target)
    # Constrain the short ends too; their closure welds then average identical
    # authored positions instead of twisting a band into a corkscrew.
    for path_id, ids in (
        (key[1], values)
        for key, values in pattern_mesh.path_vertices.items()
        if key[0] == panel_id and key[1] not in {attach_path_id, far_path_id}
    ):
        del path_id
        for global_vertex in ids:
            local_index = int(global_vertex) - start
            x, y = mesh.material_coordinates_m[local_index]
            fraction = (float(y) - attach_y) / height
            target = (1.0 - fraction) * attached_at_x(float(x)) + fraction * far_at_x(float(x))
            candidates.setdefault(local_index, []).append(target)
    constraints = _mean_targets(candidates)
    authored = _harmonic_panel_extension(
        mesh.triangles,
        mesh.vertex_count,
        constraints,
        base_positions=base,
    )
    if attachment_departure_m > 0.0:
        stitched = np.asarray(sorted(attach_constraints), dtype=np.int32)
        stitch_delta = (
            mesh.material_coordinates_m[:, None, :]
            - mesh.material_coordinates_m[stitched][None, :, :]
        )
        distance_from_stitch = np.min(
            np.linalg.norm(stitch_delta, axis=2), axis=1
        )
        cross_grain_fraction = np.clip(
            np.abs((mesh.material_coordinates_m[:, 1] - attach_y) / height),
            0.0,
            1.0,
        )
        departure = (
            float(attachment_departure_m)
            * np.sin(
                0.5
                * math.pi
                * np.clip(
                    distance_from_stitch / attachment_departure_scale_m,
                    0.0,
                    1.0,
                )
            )
            * (1.0 - cross_grain_fraction)
        )
        direction = 1.0 if far_z_offset_m >= 0.0 else -1.0
        authored[:, 2] += direction * departure
        # The actual stitch particles remain exact.  Only the compiler's
        # deliberately unstitched between-stitch guards peel away.
        authored[stitched] = np.asarray(
            [attach_constraints[int(index)] for index in stitched],
            dtype=np.float64,
        )
    positions_m[start:stop] = authored
    return len(constraints), mesh.vertex_count - len(constraints)


def _author_flared_turn_down_collar(
    pattern_mesh: GarmentPatternMesh,
    positions_m: np.ndarray,
    panel_id: str,
    *,
    attach_seam_id: str,
    attach_path_id: str,
    far_path_id: str,
    far_lift_m: float,
) -> tuple[int, int]:
    """Author a shaped collar as a lifted developable flare.

    Material lines of constant normalized width fraction fan tangentially by
    the pattern's authored flare while the cross-grain direction follows the
    outward normal of the stand rim.  Splitting the 75 mm cross-grain length
    between XY offset and a small positive lift gives a non-intersecting reset
    above the upright stand; gravity/contact may subsequently turn it down.
    """

    start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
    mesh = _panel_mesh(pattern_mesh, panel_id)
    attach_constraints = _constraints_from_host_seams(
        pattern_mesh, positions_m, panel_id, (attach_seam_id,)
    )
    attach_global = pattern_mesh.path_vertices[(panel_id, attach_path_id)]
    attach_local = attach_global - start
    attach_x = mesh.material_coordinates_m[attach_local, 0]
    attach_targets = np.asarray(
        [attach_constraints[int(index)] for index in attach_local],
        dtype=np.float64,
    )
    order = np.argsort(attach_x)
    attach_x = attach_x[order]
    attach_targets = attach_targets[order]
    attach_min, attach_max = float(attach_x[0]), float(attach_x[-1])
    attach_span = attach_max - attach_min
    if attach_span <= 1.0e-12:
        raise ValueError("A flared collar attachment has no length")

    far_global = pattern_mesh.path_vertices[(panel_id, far_path_id)]
    far_local = far_global - start
    far_coordinates = mesh.material_coordinates_m[far_local]
    far_min = float(np.min(far_coordinates[:, 0]))
    far_max = float(np.max(far_coordinates[:, 0]))
    attach_y = float(np.mean(mesh.material_coordinates_m[attach_local, 1]))
    far_y = float(np.mean(far_coordinates[:, 1]))
    material_height = far_y - attach_y
    if material_height <= 0.0:
        raise ValueError("A flared collar outer edge must lie above its attachment")
    if not 0.0 < far_lift_m < material_height:
        raise ValueError("A flared collar lift must be below its material width")
    transverse_xy_m = math.sqrt(material_height**2 - far_lift_m**2)

    def attached_at_u(u: float) -> np.ndarray:
        x = attach_min + float(np.clip(u, 0.0, 1.0)) * attach_span
        return np.asarray(
            [np.interp(x, attach_x, attach_targets[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )

    attachment_xy = attach_targets[:, :2]
    tangent_samples = np.empty_like(attachment_xy)
    tangent_samples[0] = attachment_xy[1] - attachment_xy[0]
    tangent_samples[-1] = attachment_xy[-1] - attachment_xy[-2]
    tangent_samples[1:-1] = attachment_xy[2:] - attachment_xy[:-2]
    tangent_length = np.linalg.norm(tangent_samples, axis=1)
    valid = tangent_length > 1.0e-10
    tangent_samples[valid] /= tangent_length[valid, None]
    candidate_normal = np.column_stack(
        (-tangent_samples[valid, 1], tangent_samples[valid, 0])
    )
    centroid = np.mean(attachment_xy, axis=0)
    score = float(
        np.sum(candidate_normal * (centroid[None, :] - attachment_xy[valid]))
    )
    inward_sign = 1.0 if score >= 0.0 else -1.0

    def target(x: float, y: float) -> np.ndarray:
        v = float(np.clip((y - attach_y) / material_height, 0.0, 1.0))
        left_x = (1.0 - v) * attach_min + v * far_min
        right_x = (1.0 - v) * attach_max + v * far_max
        u = float(np.clip((x - left_x) / (right_x - left_x), 0.0, 1.0))
        attached = attached_at_u(u)
        epsilon = 1.0e-3
        before = attached_at_u(max(0.0, u - epsilon))[:2]
        after = attached_at_u(min(1.0, u + epsilon))[:2]
        tangent = after - before
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1.0e-10:
            raise ValueError("A flared collar attachment has a degenerate tangent")
        tangent /= tangent_norm
        outward = -inward_sign * np.asarray((-tangent[1], tangent[0]))
        bottom_x = attach_min + u * attach_span
        tangential_flare = float(x - bottom_x)
        return attached + np.asarray(
            (
                tangential_flare * tangent[0] + v * transverse_xy_m * outward[0],
                tangential_flare * tangent[1] + v * transverse_xy_m * outward[1],
                v * far_lift_m,
            )
        )

    base = np.asarray(
        [target(float(x), float(y)) for x, y in mesh.material_coordinates_m],
        dtype=np.float64,
    )
    candidates: dict[int, list[np.ndarray]] = {
        index: [value.copy()] for index, value in attach_constraints.items()
    }
    for (owner, _path_id), path_vertices in pattern_mesh.path_vertices.items():
        if owner != panel_id:
            continue
        for global_vertex in path_vertices:
            local_index = int(global_vertex) - start
            x, y = mesh.material_coordinates_m[local_index]
            candidates.setdefault(local_index, []).append(
                target(float(x), float(y))
            )
    constraints = _mean_targets(candidates)
    positions_m[start:stop] = _harmonic_panel_extension(
        mesh.triangles,
        mesh.vertex_count,
        constraints,
        base_positions=base,
    )
    return len(constraints), mesh.vertex_count - len(constraints)


def _finish_flat_lay_strategy(
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
    positions_m: np.ndarray,
    *,
    strategy_id: str,
    root_panel_id: str,
    body_layer_separation_m: float,
    constrained_counts: Mapping[str, int],
    harmonic_counts: Mapping[str, int],
    notes: tuple[str, ...],
) -> FlatLaySeedResult:
    if not np.all(np.isfinite(positions_m)):
        missing = np.flatnonzero(~np.all(np.isfinite(positions_m), axis=1))
        raise RuntimeError(
            f"Flat-lay strategy {strategy_id!r} left {len(missing)} vertices unauthored"
        )
    centre = np.average(
        positions_m, axis=0, weights=pattern_mesh.vertex_mass_kg
    )
    positions = positions_m - centre
    seam_rms, seam_maximum = _seam_gap_metrics(
        construction, pattern_mesh, positions
    )
    poses = MappingProxyType(
        {
            panel.id: RigidPanelSeedPose(
                panel_id=panel.id,
                rotation_rad=0.0,
                translation_m=(0.0, 0.0),
                layer_offset_m=float(
                    np.median(
                        positions[
                            slice(*pattern_mesh.panel_vertex_ranges[panel.id]), 2
                        ]
                    )
                ),
                mirror_material_x=panel.seed_mirror_material_x,
            )
            for panel in construction.panels
        }
    )
    return FlatLaySeedResult(
        seed=SeedAssembly(
            construction_id=construction.id,
            material_vertex_positions_m=positions,
            panel_poses=poses,
            root_panel_id=root_panel_id,
            inter_panel_seam_rms_gap_m=seam_rms,
            maximum_inter_panel_seam_rms_gap_m=seam_maximum,
        ),
        report=FlatLaySeedReport(
            construction_id=construction.id,
            strategy_id=strategy_id,
            reset_only_numerical_preconditioner=True,
            body_layer_separation_m=body_layer_separation_m,
            constrained_vertex_count_by_panel=MappingProxyType(
                dict(constrained_counts)
            ),
            harmonic_vertex_count_by_panel=MappingProxyType(
                dict(harmonic_counts)
            ),
            inter_panel_seam_rms_gap_m=seam_rms,
            maximum_inter_panel_seam_rms_gap_m=seam_maximum,
            notes=notes,
        ),
    )


def _tee_flat_lay_seed(
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
    config: FlatLayResetConfig,
) -> FlatLaySeedResult:
    required_panels = {"front", "back", "left_sleeve", "right_sleeve", "neck_rib"}
    if set(pattern_mesh.panel_ids) != required_panels:
        raise ValueError("The T-shirt flat-lay strategy received an unexpected panel set")

    positions = np.full((pattern_mesh.vertex_count, 3), np.nan, dtype=np.float64)
    half_height = 0.5 * config.body_layer_separation_m
    for panel_id, height in (("front", half_height), ("back", -half_height)):
        start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
        positions[start:stop, :2] = pattern_mesh.material_coordinates_m[start:stop]
        positions[start:stop, 2] = height

    stitch_by_id = {stitch.seam_id: stitch for stitch in pattern_mesh.stitches}
    constrained_counts: dict[str, int] = {"front": 0, "back": 0}
    harmonic_counts: dict[str, int] = {"front": 0, "back": 0}

    for sleeve_id, side_sign, cap_seams in (
        (
            "left_sleeve",
            -1.0,
            ("left_sleeve_front_cap", "left_sleeve_back_cap"),
        ),
        (
            "right_sleeve",
            1.0,
            ("right_sleeve_front_cap", "right_sleeve_back_cap"),
        ),
    ):
        start, stop = pattern_mesh.panel_vertex_ranges[sleeve_id]
        panel_mesh = pattern_mesh.panel_meshes[
            pattern_mesh.panel_ids.index(sleeve_id)
        ]
        target_candidates: dict[int, list[np.ndarray]] = {}
        cap_endpoint_vertices = {
            int(pattern_mesh.path_vertices[(sleeve_id, path_id)][endpoint])
            for path_id in ("cap_front", "cap_back")
            for endpoint in (0, -1)
        }

        def add(global_vertex: int, target: np.ndarray) -> None:
            local_vertex = int(global_vertex) - start
            if not 0 <= local_vertex < stop - start:
                raise ValueError(f"A target leaves sleeve {sleeve_id!r}")
            target_candidates.setdefault(local_vertex, []).append(
                np.asarray(target, dtype=np.float64)
            )

        # Cap samples are copied from the corresponding body cut edge.  This
        # exactly respects the compiled ease/stitch correspondence.
        for seam_id in cap_seams:
            for first_raw, second_raw in stitch_by_id[seam_id].vertex_pairs:
                first, second = int(first_raw), int(second_raw)
                if start <= first < stop:
                    sleeve_vertex, body_vertex = first, second
                elif start <= second < stop:
                    sleeve_vertex, body_vertex = second, first
                else:
                    raise ValueError(f"Cap seam {seam_id!r} misses its sleeve")
                target = positions[body_vertex].copy()
                if sleeve_vertex in cap_endpoint_vertices:
                    target[2] = 0.0
                add(sleeve_vertex, target)

        # The flattened tube is constructed isometrically at the open cuff:
        # S is the underarm-seam cuff point, F the outer fold point, and the
        # cuff half-width S--F is orthogonal to the longitudinal fold C--F.
        panel = construction.panel_map[sleeve_id]
        cap_front = panel.path("cap_front")
        crown_xy = np.asarray(cap_front.points_m[-1], dtype=np.float64)
        # Body and sleeve cap targets use the same global frame; read the exact
        # constrained endpoints rather than relying on panel dimensions.
        crown_global = int(
            pattern_mesh.path_vertices[(sleeve_id, "cap_front")][-1]
        )
        armpit_global = int(
            pattern_mesh.path_vertices[(sleeve_id, "cap_front")][0]
        )
        crown = np.mean(target_candidates[crown_global - start], axis=0)[:2]
        armpit = np.mean(target_candidates[armpit_global - start], axis=0)[:2]
        del crown_xy  # the material landmark is retained by the compiled path
        cap_chord = armpit - crown
        cap_chord_length = float(np.linalg.norm(cap_chord))
        cuff_half_width = 0.5 * panel.path("sleeve_hem").length_m
        if not 0.0 < cuff_half_width < cap_chord_length:
            raise ValueError("T-shirt cuff width is incompatible with its armhole chord")
        cap_direction = cap_chord / cap_chord_length
        width_ratio = cuff_half_width / cap_chord_length
        cap_normal = np.asarray((-cap_direction[1], cap_direction[0]))
        cuff_normal = (
            -width_ratio * cap_direction
            + math.sqrt(1.0 - width_ratio**2) * side_sign * cap_normal
        )
        longitudinal = side_sign * np.asarray(
            (cuff_normal[1], -cuff_normal[0])
        )
        underarm_length = 0.5 * (
            panel.path("underarm_front").length_m
            + panel.path("underarm_back").length_m
        )
        cuff_seam = armpit + underarm_length * longitudinal
        cuff_fold = cuff_seam + cuff_half_width * cuff_normal

        def add_path(path_id: str, target_at_fraction) -> None:
            ids = pattern_mesh.path_vertices[(sleeve_id, path_id)]
            fractions = pattern_mesh.path_fractions[(sleeve_id, path_id)]
            for vertex, fraction in zip(ids, fractions):
                add(int(vertex), target_at_fraction(float(fraction)))

        add_path(
            "sleeve_hem",
            lambda fraction: np.asarray(
                (
                    *(
                        cuff_seam
                        + (2.0 * fraction if fraction <= 0.5 else 2.0 * (1.0 - fraction))
                        * (cuff_fold - cuff_seam)
                    ),
                    -half_height * math.sin(2.0 * math.pi * fraction),
                )
            ),
        )
        add_path(
            "underarm_front",
            lambda fraction: np.asarray(
                (
                    *(cuff_seam + fraction * (armpit - cuff_seam)),
                    0.0,
                )
            ),
        )
        add_path(
            "underarm_back",
            lambda fraction: np.asarray(
                (
                    *(armpit + fraction * (cuff_seam - armpit)),
                    0.0,
                )
            ),
        )
        add_path(
            "flat_lay_fold_axis",
            lambda fraction: np.asarray(
                (*(cuff_fold + fraction * (crown - cuff_fold)), 0.0)
            ),
        )
        constraints = _mean_targets(target_candidates)
        local_material = panel_mesh.material_coordinates_m
        fold_path = panel.path("flat_lay_fold_axis")
        fold_start = np.asarray(fold_path.points_m[0], dtype=np.float64)
        fold_end = np.asarray(fold_path.points_m[-1], dtype=np.float64)
        fold_delta = fold_end - fold_start
        fold_length_squared = float(np.dot(fold_delta, fold_delta))
        if fold_length_squared <= 1.0e-12:
            raise ValueError("A sleeve flat-lay fold axis cannot be degenerate")
        fold_fraction = (
            (local_material - fold_start[None, :]) @ fold_delta
        ) / fold_length_squared
        sleeve_base = np.zeros((panel_mesh.vertex_count, 3), dtype=np.float64)
        sleeve_base[:, :2] = (
            cuff_fold[None, :]
            + fold_fraction[:, None] * (crown - cuff_fold)[None, :]
            - np.abs(local_material[:, 0, None]) * cuff_normal[None, :]
        )
        sleeve_positions = _harmonic_panel_extension(
            panel_mesh.triangles,
            panel_mesh.vertex_count,
            constraints,
            base_positions=sleeve_base,
        )
        positions[start:stop] = sleeve_positions
        constrained_counts[sleeve_id] = len(constraints)
        harmonic_counts[sleeve_id] = panel_mesh.vertex_count - len(constraints)

    rib_id = "neck_rib"
    rib_start, rib_stop = pattern_mesh.panel_vertex_ranges[rib_id]
    rib_mesh = pattern_mesh.panel_meshes[pattern_mesh.panel_ids.index(rib_id)]
    rib_candidates: dict[int, list[np.ndarray]] = {}

    def add_rib(global_vertex: int, target: np.ndarray) -> None:
        local_vertex = int(global_vertex) - rib_start
        if not 0 <= local_vertex < rib_stop - rib_start:
            raise ValueError("A neck-rib target leaves the rib panel")
        rib_candidates.setdefault(local_vertex, []).append(
            np.asarray(target, dtype=np.float64)
        )

    for seam_id in ("front_neck_rib", "back_neck_rib"):
        for first_raw, second_raw in stitch_by_id[seam_id].vertex_pairs:
            first, second = int(first_raw), int(second_raw)
            if rib_start <= first < rib_stop:
                rib_vertex, body_vertex = first, second
            elif rib_start <= second < rib_stop:
                rib_vertex, body_vertex = second, first
            else:
                raise ValueError(f"Neck seam {seam_id!r} misses the rib panel")
            add_rib(rib_vertex, positions[body_vertex])

    attach_global = pattern_mesh.path_vertices[(rib_id, "neck_attach")]
    attach_local = _panel_local_indices(pattern_mesh, rib_id, attach_global)
    attach_targets = _mean_targets(rib_candidates)
    if any(int(index) not in attach_targets for index in attach_local):
        raise ValueError("Compiled neck stitches do not constrain the whole attach edge")
    attach_material_x = pattern_mesh.material_coordinates_m[attach_global, 0]
    order = np.argsort(attach_material_x)
    attach_material_x = attach_material_x[order]
    attach_positions = np.asarray(
        [attach_targets[int(index)] for index in attach_local[order]],
        dtype=np.float64,
    )
    rib_width = construction.panel_map[rib_id].path("closure_right").length_m
    neck_centre = np.asarray(
        (
            0.5 * (float(np.min(attach_positions[:, 0])) + float(np.max(attach_positions[:, 0]))),
            float(np.max(attach_positions[:, 1]))
            + config.neck_opening_center_offset_fraction * rib_width,
        ),
        dtype=np.float64,
    )

    opening_global = pattern_mesh.path_vertices[(rib_id, "neck_opening")]
    opening_targets: dict[int, np.ndarray] = {}
    for global_vertex in opening_global:
        material_x = float(pattern_mesh.material_coordinates_m[int(global_vertex), 0])
        attach = np.asarray(
            [
                np.interp(material_x, attach_material_x, attach_positions[:, axis])
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        inward = neck_centre - attach[:2]
        norm = float(np.linalg.norm(inward))
        if norm <= 1.0e-12:
            raise ValueError("Neck opening offset has an undefined inward normal")
        target = attach.copy()
        target[:2] += rib_width * inward / norm
        opening_targets[int(global_vertex)] = target
        add_rib(int(global_vertex), target)

    for path_id in ("closure_right", "closure_left"):
        ids = pattern_mesh.path_vertices[(rib_id, path_id)]
        fractions = pattern_mesh.path_fractions[(rib_id, path_id)]
        start_target = (
            attach_targets[int(ids[0] - rib_start)]
            if int(ids[0] - rib_start) in attach_targets
            else opening_targets[int(ids[0])]
        )
        end_target = (
            attach_targets[int(ids[-1] - rib_start)]
            if int(ids[-1] - rib_start) in attach_targets
            else opening_targets[int(ids[-1])]
        )
        for vertex, fraction in zip(ids, fractions):
            add_rib(
                int(vertex),
                (1.0 - float(fraction)) * start_target
                + float(fraction) * end_target,
            )

    rib_constraints = _mean_targets(rib_candidates)
    positions[rib_start:rib_stop] = _harmonic_panel_extension(
        rib_mesh.triangles,
        rib_mesh.vertex_count,
        rib_constraints,
    )
    constrained_counts[rib_id] = len(rib_constraints)
    harmonic_counts[rib_id] = rib_mesh.vertex_count - len(rib_constraints)

    if not np.all(np.isfinite(positions)):
        raise RuntimeError("T-shirt flat-lay strategy left unauthored vertices")
    centre = np.average(positions, axis=0, weights=pattern_mesh.vertex_mass_kg)
    positions -= centre
    seam_rms, seam_maximum = _seam_gap_metrics(
        construction,
        pattern_mesh,
        positions,
    )
    poses = MappingProxyType(
        {
            panel.id: RigidPanelSeedPose(
                panel_id=panel.id,
                rotation_rad=0.0,
                translation_m=(0.0, 0.0),
                layer_offset_m=(
                    half_height
                    if panel.id == "front"
                    else -half_height
                    if panel.id == "back"
                    else 0.0
                ),
                mirror_material_x=panel.seed_mirror_material_x,
            )
            for panel in construction.panels
        }
    )
    seed = SeedAssembly(
        construction_id=construction.id,
        material_vertex_positions_m=positions,
        panel_poses=poses,
        root_panel_id="front",
        inter_panel_seam_rms_gap_m=seam_rms,
        maximum_inter_panel_seam_rms_gap_m=seam_maximum,
    )
    report = FlatLaySeedReport(
        construction_id=construction.id,
        strategy_id="sewn_tee_flattened_shell_v1",
        reset_only_numerical_preconditioner=True,
        body_layer_separation_m=config.body_layer_separation_m,
        constrained_vertex_count_by_panel=MappingProxyType(constrained_counts),
        harmonic_vertex_count_by_panel=MappingProxyType(harmonic_counts),
        inter_panel_seam_rms_gap_m=seam_rms,
        maximum_inter_panel_seam_rms_gap_m=seam_maximum,
        notes=(
            "Body front and back are separated layers joined by welded construction seams.",
            "Sleeves are open-cuff flattened tubes with cap, underarm, cuff, and fold-axis constraints.",
            "The neck rib follows its stitched attachment and offsets toward the neck opening.",
            "Layer separation is reset-only numerical preconditioning, not calibrated textile thickness.",
        ),
    )
    return FlatLaySeedResult(seed=seed, report=report)


def _woven_shirt_flat_lay_seed(
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
    config: FlatLayResetConfig,
) -> FlatLaySeedResult:
    required = {
        "left_front",
        "right_front",
        "back",
        "left_sleeve",
        "right_sleeve",
        "left_cuff",
        "right_cuff",
        "back_yoke",
        "chest_pocket",
        "collar_stand",
        "collar",
    }
    if set(pattern_mesh.panel_ids) != required:
        raise ValueError("The woven-shirt flat lay received an unexpected panel set")
    positions = np.full((pattern_mesh.vertex_count, 3), np.nan, dtype=np.float64)
    half_height = 0.5 * config.body_layer_separation_m
    placket_half_gap_m = 1.0e-3
    for panel_id, x_sign, height in (
        ("left_front", -1.0, half_height),
        ("right_front", 1.0, half_height),
        ("back", 1.0, -half_height),
    ):
        start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
        positions[start:stop, :2] = pattern_mesh.material_coordinates_m[start:stop]
        positions[start:stop, 0] *= x_sign
        if panel_id != "back":
            # The reset depicts an unbuttoned shirt.  Distinct placket cut
            # edges therefore have a narrow, visible opening; leaving them at
            # the same coordinates created 138 exact coplanar intersections
            # and falsely represented two separate front edges as one seam.
            positions[start:stop, 0] += x_sign * placket_half_gap_m
        positions[start:stop, 2] = height

    constrained: dict[str, int] = {
        "left_front": 0,
        "right_front": 0,
        "back": 0,
    }
    harmonic: dict[str, int] = {
        "left_front": 0,
        "right_front": 0,
        "back": 0,
    }
    for sleeve_id, side_sign, seams in (
        ("left_sleeve", -1.0, ("left_cap_front", "left_cap_back")),
        ("right_sleeve", 1.0, ("right_cap_front", "right_cap_back")),
    ):
        constrained[sleeve_id], harmonic[sleeve_id] = _author_flattened_sleeve(
            construction,
            pattern_mesh,
            positions,
            sleeve_id,
            side_sign=side_sign,
            cap_seam_ids=seams,
            opening_path_id="wrist",
            half_height=half_height,
        )

    for sleeve_id, cuff_id, seam_id in (
        ("left_sleeve", "left_cuff", "left_cuff_attach"),
        ("right_sleeve", "right_cuff", "right_cuff_attach"),
    ):
        cap_vertices = np.concatenate(
            (
                pattern_mesh.path_vertices[(sleeve_id, "cap_front")],
                pattern_mesh.path_vertices[(sleeve_id, "cap_back")],
            )
        )
        wrist_vertices = pattern_mesh.path_vertices[(sleeve_id, "wrist")]
        direction = (
            np.mean(positions[wrist_vertices], axis=0)
            - np.mean(positions[cap_vertices], axis=0)
        )
        direction[2] = 0.0
        direction /= np.linalg.norm(direction)
        constrained[cuff_id], harmonic[cuff_id] = _author_rectangular_band(
            pattern_mesh,
            positions,
            cuff_id,
            attach_seam_ids=(seam_id,),
            attach_path_id="sleeve_attach",
            far_path_id="wrist_opening",
            far_offset_vector_m=tuple(
                map(
                    float,
                    0.060 * direction + np.asarray((0.0, 0.0, 0.4e-3)),
                )
            ),
        )

    constrained["back_yoke"], harmonic["back_yoke"] = _author_attached_panel(
        pattern_mesh,
        positions,
        "back_yoke",
        ("yoke_bottom", "yoke_right", "yoke_top", "yoke_left"),
        base_z_offset_m=-2.0e-3,
    )
    constrained["chest_pocket"], harmonic["chest_pocket"] = _author_attached_panel(
        pattern_mesh,
        positions,
        "chest_pocket",
        ("pocket_bottom", "pocket_right", "pocket_left"),
        base_z_offset_m=2.0e-3,
    )
    constrained["collar_stand"], harmonic["collar_stand"] = _author_rectangular_band(
        pattern_mesh,
        positions,
        "collar_stand",
        attach_seam_ids=("left_neck_stand", "back_neck_stand", "right_neck_stand"),
        attach_path_id="neck_attach",
        far_path_id="collar_attach",
        # A 35 mm planar inward offset exceeds the local radius of the neck
        # curve and must cusp.  A real interfaced stand rolls out of the shirt
        # plane; using its actual cross-grain width vertically preserves the
        # strip metric and gives the turn-down collar a valid attachment rim.
        far_offset_vector_m=(0.0, 0.0, 0.035),
    )
    constrained["collar"], harmonic["collar"] = _author_flared_turn_down_collar(
        pattern_mesh,
        positions,
        "collar",
        attach_seam_id="collar_to_stand",
        attach_path_id="stand_attach",
        far_path_id="outer_edge",
        far_lift_m=0.020,
    )
    return _finish_flat_lay_strategy(
        construction,
        pattern_mesh,
        positions,
        strategy_id="woven_shirt_front_flattened_shell_v1",
        root_panel_id="left_front",
        body_layer_separation_m=config.body_layer_separation_m,
        constrained_counts=constrained,
        harmonic_counts=harmonic,
        notes=(
            "Front halves form a continuous open placket above a separated back layer.",
            "Long sleeves and cuffs are authored as open-ended flattened tubes.",
            "The physical patch pocket, doubled yoke, collar stand, and turn-down collar retain their own panels.",
            "All visual landmarks come from compiled stitch correspondences; no shirt silhouette substitute is used.",
        ),
    )


def _terry_towel_flat_lay_seed(
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
    config: FlatLayResetConfig,
) -> FlatLaySeedResult:
    del config
    if pattern_mesh.panel_ids != ("terry_body",):
        raise ValueError("The terry-towel flat lay received an unexpected panel set")
    positions = np.column_stack(
        (
            pattern_mesh.material_coordinates_m,
            np.zeros(pattern_mesh.vertex_count, dtype=np.float64),
        )
    )
    return _finish_flat_lay_strategy(
        construction,
        pattern_mesh,
        positions,
        strategy_id="terry_rectangular_sheet_isometric_v1",
        root_panel_id="terry_body",
        body_layer_separation_m=0.0,
        constrained_counts={"terry_body": pattern_mesh.vertex_count},
        harmonic_counts={"terry_body": 0},
        notes=(
            "The towel is one physical rectangular sheet, not a garment-shell surrogate.",
            "Four multilayer hems and both dobby bands remain construction features on the real panel.",
        ),
    )


def _denim_jeans_flat_lay_seed(
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
    config: FlatLayResetConfig,
) -> FlatLaySeedResult:
    required = {
        "front_left",
        "front_right",
        "back_left",
        "back_right",
        "front_left_waistband",
        "front_right_waistband",
        "back_left_waistband",
        "back_right_waistband",
        "left_pocket_outer",
        "left_pocket_inner",
        "right_pocket_outer",
        "right_pocket_inner",
        "left_back_patch_pocket",
        "right_back_patch_pocket",
        "coin_pocket",
        "back_left_yoke",
        "back_right_yoke",
        "fly_shield",
        "fly_facing",
    }
    if set(pattern_mesh.panel_ids) != required:
        raise ValueError("The jeans flat lay received an unexpected panel set")
    positions = np.full((pattern_mesh.vertex_count, 3), np.nan, dtype=np.float64)
    half_height = 0.5 * config.body_layer_separation_m
    constrained: dict[str, int] = {}
    harmonic: dict[str, int] = {}

    leg_ids = ("front_left", "front_right", "back_left", "back_right")
    tube_top_y_m = 0.70

    def boundary_x_at_y(panel_id: str, path_id: str, y: float) -> float:
        points = np.asarray(
            construction.panel_map[panel_id].path(path_id).points_m,
            dtype=np.float64,
        )
        hits: list[float] = []
        for first, second in zip(points[:-1], points[1:]):
            low, high = sorted((float(first[1]), float(second[1])))
            if low - 1.0e-10 <= y <= high + 1.0e-10:
                dy = float(second[1] - first[1])
                if abs(dy) <= 1.0e-12:
                    hits.extend((float(first[0]), float(second[0])))
                else:
                    fraction = (y - float(first[1])) / dy
                    hits.append(float(first[0] + fraction * (second[0] - first[0])))
        if not hits:
            raise ValueError(
                f"Jeans boundary {panel_id!r}/{path_id!r} misses y={y:.6f}"
            )
        return float(np.mean(hits))

    def outline_x_bounds(panel_id: str, y: float) -> tuple[float, float]:
        hits: list[float] = []
        panel = construction.panel_map[panel_id]
        for path in panel.outline_paths:
            points = np.asarray(path.points_m, dtype=np.float64)
            for first, second in zip(points[:-1], points[1:]):
                low, high = sorted((float(first[1]), float(second[1])))
                if low - 1.0e-10 <= y <= high + 1.0e-10:
                    dy = float(second[1] - first[1])
                    if abs(dy) <= 1.0e-12:
                        hits.extend((float(first[0]), float(second[0])))
                    else:
                        fraction = (y - float(first[1])) / dy
                        hits.append(
                            float(first[0] + fraction * (second[0] - first[0]))
                        )
        if len(hits) < 2:
            coordinates = _panel_mesh(
                pattern_mesh, panel_id
            ).material_coordinates_m[:, 0]
            return float(np.min(coordinates)), float(np.max(coordinates))
        return float(np.min(hits)), float(np.max(hits))

    def half_ellipse_table(
        semi_major: float,
        arc_length: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if arc_length <= 2.0 * semi_major:
            raise ValueError("A jeans tube arc cannot be shorter than its chord")
        angles = np.linspace(0.0, math.pi, 257)

        def length(semi_minor: float) -> float:
            speed = np.sqrt(
                semi_major**2 * np.sin(angles) ** 2
                + semi_minor**2 * np.cos(angles) ** 2
            )
            return float(np.trapezoid(speed, angles))

        low, high = 0.0, arc_length
        for _ in range(64):
            middle = 0.5 * (low + high)
            if length(middle) < arc_length:
                low = middle
            else:
                high = middle
        semi_minor = 0.5 * (low + high)
        speed = np.sqrt(
            semi_major**2 * np.sin(angles) ** 2
            + semi_minor**2 * np.cos(angles) ** 2
        )
        cumulative = np.zeros_like(angles)
        delta = angles[1] - angles[0]
        cumulative[1:] = np.cumsum(0.5 * (speed[:-1] + speed[1:]) * delta)
        cumulative /= cumulative[-1]
        return semi_minor, cumulative, angles

    front_width_at_crotch = boundary_x_at_y(
        "front_left", "inseam", tube_top_y_m
    )
    back_width_at_crotch = boundary_x_at_y(
        "back_left", "inseam", tube_top_y_m
    )
    crotch_semi_major = 0.48 * min(
        front_width_at_crotch, back_width_at_crotch
    )

    def leg_center_abs(y: float) -> float:
        fraction = float(np.clip(y / tube_top_y_m, 0.0, 1.0))
        return (1.0 - fraction) * 0.100 + fraction * crotch_semi_major

    bases: dict[str, np.ndarray] = {}
    for panel_id in leg_ids:
        mesh = _panel_mesh(pattern_mesh, panel_id)
        front = panel_id.startswith("front_")
        side = panel_id.rsplit("_", 1)[1]
        side_sign = -1.0 if side == "left" else 1.0
        layer_sign = 1.0 if front else -1.0
        base = np.empty((mesh.vertex_count, 3), dtype=np.float64)
        for index, (x_raw, y_raw) in enumerate(mesh.material_coordinates_m):
            x, y = float(x_raw), float(y_raw)
            if y <= tube_top_y_m + 1.0e-10:
                front_width = boundary_x_at_y(
                    f"front_{side}", "inseam", min(y, tube_top_y_m)
                )
                back_width = boundary_x_at_y(
                    f"back_{side}", "inseam", min(y, tube_top_y_m)
                )
                semi_major = 0.48 * min(front_width, back_width)
                material_width = front_width if front else back_width
                semi_minor, cumulative, angles = half_ellipse_table(
                    semi_major, material_width
                )
                arclength_fraction = float(
                    np.clip(x / material_width, 0.0, 1.0)
                )
                angle = float(
                    np.interp(arclength_fraction, cumulative, angles)
                )
                base[index] = (
                    side_sign
                    * (leg_center_abs(y) + semi_major * math.cos(angle)),
                    y,
                    layer_sign * semi_minor * math.sin(angle),
                )
            else:
                # The upper four panels form a second flattened shell: each
                # front/back half runs from a centre seam to the shared side
                # seam along its own arclength-preserving bowed cross-section.
                # This retains the genuine sloped waist instead of stretching
                # every horizontal slice to a fixed hip width.
                own_min, own_max = outline_x_bounds(panel_id, y)
                own_width = own_max - own_min
                if own_width <= 1.0e-9:
                    base[index] = (0.0, y, layer_sign * half_height)
                    continue
                upper_fraction = float(
                    np.clip((y - tube_top_y_m) / 0.080, 0.0, 1.0)
                )
                # Blend continuously from the elliptical leg tube at the
                # crotch into an almost straight waist shell.  Jumping from a
                # 28 mm-deep back arc at y=0.700 m to a planar row immediately
                # above it created a corrugated pocket host and extreme
                # longitudinal strain.
                center_height = min(
                    half_height * upper_fraction,
                    0.5 * own_width,
                )
                front_min, front_max = outline_x_bounds(f"front_{side}", y)
                back_min, back_max = outline_x_bounds(f"back_{side}", y)
                lower_chord = 0.96 * min(
                    max(front_max - front_min, 1.0e-6),
                    max(back_max - back_min, 1.0e-6),
                )
                straight_chord = math.sqrt(
                    max(own_width**2 - center_height**2, 0.0)
                )
                flatten = float(np.clip((y - tube_top_y_m) / 0.150, 0.0, 1.0))
                flatten = flatten * flatten * (3.0 - 2.0 * flatten)
                chord = min(
                    (1.0 - flatten) * lower_chord
                    + flatten * straight_chord,
                    straight_chord,
                )
                parameters = np.linspace(0.0, 1.0, 129)

                def cross_section_length(amplitude: float) -> float:
                    derivative_z = (
                        center_height
                        + amplitude * math.pi * np.cos(math.pi * parameters)
                    )
                    return float(
                        np.trapezoid(
                            np.sqrt(chord**2 + derivative_z**2), parameters
                        )
                    )

                low, high = 0.0, own_width
                if cross_section_length(0.0) >= own_width - 1.0e-10:
                    amplitude = 0.0
                else:
                    for _ in range(48):
                        middle = 0.5 * (low + high)
                        if cross_section_length(middle) < own_width:
                            low = middle
                        else:
                            high = middle
                    amplitude = 0.5 * (low + high)
                derivative_z = (
                    center_height
                    + amplitude * math.pi * np.cos(math.pi * parameters)
                )
                speed = np.sqrt(chord**2 + derivative_z**2)
                cumulative = np.zeros_like(parameters)
                parameter_delta = parameters[1] - parameters[0]
                cumulative[1:] = np.cumsum(
                    0.5 * (speed[:-1] + speed[1:]) * parameter_delta
                )
                cumulative /= cumulative[-1]
                material_fraction = float(
                    np.clip((x - own_min) / own_width, 0.0, 1.0)
                )
                parameter = float(
                    np.interp(material_fraction, cumulative, parameters)
                )
                base[index] = (
                    side_sign * chord * (1.0 - parameter),
                    y,
                    layer_sign
                    * (
                        center_height * parameter
                        + amplitude * math.sin(math.pi * parameter)
                    ),
                )
                if front:
                    fly_gate = float(np.clip((y - 0.860) / 0.060, 0.0, 1.0))
                    base[index, 0] += side_sign * 1.5e-3 * fly_gate
        bases[panel_id] = base
        start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
        positions[start:stop] = base

    candidates_by_panel: dict[str, dict[int, list[np.ndarray]]] = {
        panel_id: {} for panel_id in leg_ids
    }
    for panel_id in leg_ids:
        start, _ = pattern_mesh.panel_vertex_ranges[panel_id]
        mesh = _panel_mesh(pattern_mesh, panel_id)
        for local_index in range(mesh.vertex_count):
            candidates_by_panel[panel_id][local_index] = [
                bases[panel_id][local_index].copy()
            ]

    stitch_by_id = {stitch.seam_id: stitch for stitch in pattern_mesh.stitches}
    structural_seams = (
        "left_outseam",
        "left_inseam",
        "right_outseam",
        "right_inseam",
        "front_crotch_curve",
        "front_lower_rise",
        "back_crotch_curve",
        "back_rise",
    )
    panel_ranges = pattern_mesh.panel_vertex_ranges
    for seam_id in structural_seams:
        center_seam = seam_id.startswith(("front_", "back_"))
        center_layer_sign = 1.0 if seam_id.startswith("front_") else -1.0
        for first_raw, second_raw in stitch_by_id[seam_id].vertex_pairs:
            first, second = int(first_raw), int(second_raw)
            owners = [
                panel_id
                for panel_id in leg_ids
                if panel_ranges[panel_id][0] <= first < panel_ranges[panel_id][1]
                or panel_ranges[panel_id][0] <= second < panel_ranges[panel_id][1]
            ]
            if len(owners) != 2:
                raise ValueError(f"Jeans seam {seam_id!r} misses two leg panels")
            target = 0.5 * (positions[first] + positions[second])
            if center_seam:
                target[0] = 0.0
                rise_fraction = float(
                    np.clip((target[1] - tube_top_y_m) / 0.080, 0.0, 1.0)
                )
                target[2] = center_layer_sign * half_height * rise_fraction
            else:
                target[2] = 0.0
            for global_vertex, panel_id in ((first, owners[0]), (second, owners[1])):
                start, stop = panel_ranges[panel_id]
                if not start <= global_vertex < stop:
                    panel_id = owners[1] if panel_id == owners[0] else owners[0]
                    start, stop = panel_ranges[panel_id]
                candidates_by_panel[panel_id][global_vertex - start] = [
                    target.copy()
                ]

    for panel_id in leg_ids:
        start, stop = panel_ranges[panel_id]
        mesh = _panel_mesh(pattern_mesh, panel_id)
        constraints = _mean_targets(candidates_by_panel[panel_id])
        positions[start:stop] = _harmonic_panel_extension(
            mesh.triangles,
            mesh.vertex_count,
            constraints,
            base_positions=bases[panel_id],
        )
        constrained[panel_id] = len(constraints)
        harmonic[panel_id] = mesh.vertex_count - len(constraints)

    for leg_id in ("front_left", "front_right", "back_left", "back_right"):
        waistband_id = f"{leg_id}_waistband"
        constrained[waistband_id], harmonic[waistband_id] = _author_rectangular_band(
            pattern_mesh,
            positions,
            waistband_id,
            attach_seam_ids=(f"{leg_id}_waistband_attach",),
            attach_path_id="body_attach",
            far_path_id="waist_opening",
            far_offset_vector_m=(0.0, 0.042, 0.0),
        )

    # Pocket bags hang beneath their slanted mouths and remain distinct inner
    # and outer physical layers rather than being painted onto the jeans.
    for side in ("left", "right"):
        inward_x = 0.015 if side == "left" else -0.015
        outer_id = f"{side}_pocket_outer"
        constrained[outer_id], harmonic[outer_id] = _author_rectangular_band(
            pattern_mesh,
            positions,
            outer_id,
            attach_seam_ids=(f"{side}_pocket_opening_attach",),
            attach_path_id="mouth",
            far_path_id="bottom",
            far_offset_vector_m=(inward_x, -0.16, -1.8e-3),
        )
        inner_id = f"{side}_pocket_inner"
        outer_start, outer_stop = pattern_mesh.panel_vertex_ranges[outer_id]
        constrained[inner_id], harmonic[inner_id] = _author_attached_panel(
            pattern_mesh,
            positions,
            inner_id,
            (
                f"{side}_pocket_inner_opening_attach",
                f"{side}_pocket_bag_bottom",
                f"{side}_pocket_bag_right",
                f"{side}_pocket_bag_left",
            ),
            base_positions=positions[outer_start:outer_stop].copy(),
            base_z_offset_m=-0.2e-3,
        )

    for side in ("left", "right"):
        pocket_id = f"{side}_back_patch_pocket"
        host_id = f"back_{side}"
        start, stop = pattern_mesh.panel_vertex_ranges[pocket_id]
        pocket_mesh = _panel_mesh(pattern_mesh, pocket_id)
        host_bottom = construction.panel_map[host_id].path(
            "patch_bottom_mount"
        )
        host_origin = np.min(host_bottom.points_m, axis=0)
        host_queries = (
            pocket_mesh.material_coordinates_m + host_origin[None, :]
        )
        host_surface, _ = _interpolate_panel_surface(
            pattern_mesh,
            positions,
            host_id,
            host_queries,
        )
        seam_ids = (
            f"{side}_back_pocket_bottom",
            f"{side}_back_pocket_right",
            f"{side}_back_pocket_left",
        )
        host = _constraints_from_host_seams(
            pattern_mesh, positions, pocket_id, seam_ids
        )
        # Measure departure from the *actual numerical stitches*, not from
        # the drawn U-shaped cut edge.  The mesher deliberately inserts an
        # unstitched guard between adjacent stitch particles so a mounted
        # layer can peel away instead of sharing a zero-thickness triangle
        # strip with its host.  Treating every boundary guard as stitched was
        # the source of the former exact host/pocket intersections.
        stitched = np.asarray(sorted(host), dtype=np.int32)
        stitch_delta = (
            pocket_mesh.material_coordinates_m[:, None, :]
            - pocket_mesh.material_coordinates_m[stitched][None, :, :]
        )
        distance_from_stitch = np.min(
            np.linalg.norm(stitch_delta, axis=2), axis=1
        )
        # A real patch pocket turns sharply away from its U topstitch and then
        # relaxes into a shallow usable cavity.  Ten millimetres is a visible,
        # plausible empty-pocket stand-off; the 8 mm departure scale keeps
        # the finite-element first row embedded without inflating the whole
        # pocket.  Both values remain construction priors, not measurements.
        lift = 10.0e-3 * np.sin(
            0.5
            * math.pi
            * np.clip(distance_from_stitch / 8.0e-3, 0.0, 1.0)
        )
        base = host_surface.copy()
        base[:, 2] -= lift
        positions[start:stop] = _harmonic_panel_extension(
            pocket_mesh.triangles,
            pocket_mesh.vertex_count,
            host,
            base_positions=base,
        )
        constrained[pocket_id] = len(host)
        harmonic[pocket_id] = pocket_mesh.vertex_count - len(host)
        yoke_id = f"back_{side}_yoke"
        constrained[yoke_id], harmonic[yoke_id] = _author_attached_panel(
            pattern_mesh,
            positions,
            yoke_id,
            tuple(f"{side}_yoke_{edge}" for edge in ("bottom", "right", "top", "left")),
            base_z_offset_m=-1.5e-3,
        )

    constrained["coin_pocket"], harmonic["coin_pocket"] = _author_attached_panel(
        pattern_mesh,
        positions,
        "coin_pocket",
        ("coin_bottom", "coin_right", "coin_left"),
        base_z_offset_m=2.0e-3,
    )
    for panel_id, seam_id, side_sign in (
        ("fly_shield", "fly_shield_attach", -1.0),
        ("fly_facing", "fly_facing_attach", 1.0),
    ):
        start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
        mesh = _panel_mesh(pattern_mesh, panel_id)
        host = _constraints_from_host_seams(
            pattern_mesh, positions, panel_id, (seam_id,)
        )
        attach_global = pattern_mesh.path_vertices[(panel_id, "rise_attach")]
        attach_local = attach_global - start
        attach_y = mesh.material_coordinates_m[attach_local, 1]
        attach_targets = np.asarray(
            [host[int(index)] for index in attach_local], dtype=np.float64
        )
        order = np.argsort(attach_y)
        attach_y, attach_targets = attach_y[order], attach_targets[order]
        attach_x = float(
            np.mean(mesh.material_coordinates_m[attach_local, 0])
        )
        piece_width = float(np.ptp(mesh.material_coordinates_m[:, 0]))
        base = np.empty((mesh.vertex_count, 3), dtype=np.float64)
        for index, (x, y) in enumerate(mesh.material_coordinates_m):
            anchor = np.asarray(
                [
                    np.interp(float(y), attach_y, attach_targets[:, axis])
                    for axis in range(3)
                ]
            )
            free_distance = float(np.clip(attach_x - x, 0.0, piece_width))
            base[index] = anchor + np.asarray(
                (
                    side_sign * free_distance,
                    0.0,
                    -2.0e-3 * free_distance / piece_width,
                )
            )
        positions[start:stop] = _harmonic_panel_extension(
            mesh.triangles,
            mesh.vertex_count,
            host,
            base_positions=base,
        )
        constrained[panel_id] = len(host)
        harmonic[panel_id] = mesh.vertex_count - len(host)

    return _finish_flat_lay_strategy(
        construction,
        pattern_mesh,
        positions,
        strategy_id="five_pocket_jeans_flattened_leg_tubes_v1",
        root_panel_id="front_left",
        body_layer_separation_m=config.body_layer_separation_m,
        constrained_counts=constrained,
        harmonic_counts=harmonic,
        notes=(
            "Each leg is an actual front/back sewn tube flattened about inseam and outseam fold axes.",
            "The four-piece waistband, two front pocket bags, two patch pockets, coin pocket, yokes, and fly pieces remain physical panels.",
            "Pocket bags are laid behind their slanted openings; no pocket is a texture decal.",
        ),
    )
def _asymmetric_top_flat_lay_seed(
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
    config: FlatLayResetConfig,
) -> FlatLaySeedResult:
    required = {
        "front_under",
        "back",
        "left_bell_sleeve",
        "right_cap_sleeve",
        "drape_overlay",
        "left_tie",
        "right_tie",
        "front_neck_facing",
        "back_neck_facing",
    }
    if set(pattern_mesh.panel_ids) != required:
        raise ValueError("The asymmetric-top flat lay received an unexpected panel set")
    positions = np.full((pattern_mesh.vertex_count, 3), np.nan, dtype=np.float64)
    half_height = 0.5 * config.body_layer_separation_m
    for panel_id, height in (("front_under", half_height), ("back", -half_height)):
        start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
        positions[start:stop, :2] = pattern_mesh.material_coordinates_m[start:stop]
        positions[start:stop, 2] = height
        # The two body layers meet at the physical neck seam.  Bringing that
        # shared rim to the mid-plane before welding prevents a facing corner
        # from being mass-averaged across the full front/back reset gap.
        positions[pattern_mesh.path_vertices[(panel_id, "neckline")], 2] = 0.0
    constrained: dict[str, int] = {"front_under": 0, "back": 0}
    harmonic: dict[str, int] = {"front_under": 0, "back": 0}
    for sleeve_id, side_sign, seams in (
        (
            "left_bell_sleeve",
            -1.0,
            ("left_bell_cap_front", "left_bell_cap_back"),
        ),
        (
            "right_cap_sleeve",
            1.0,
            ("right_cap_front", "right_cap_back"),
        ),
    ):
        constrained[sleeve_id], harmonic[sleeve_id] = _author_flattened_sleeve(
            construction,
            pattern_mesh,
            positions,
            sleeve_id,
            side_sign=side_sign,
            cap_seam_ids=seams,
            opening_path_id="opening_hem",
            half_height=half_height,
        )

    overlay_mesh = _panel_mesh(pattern_mesh, "drape_overlay")
    overlay_base = np.column_stack(
        (
            overlay_mesh.material_coordinates_m,
            np.full(overlay_mesh.vertex_count, half_height + 0.8e-3),
        )
    )
    constrained["drape_overlay"], harmonic["drape_overlay"] = _author_attached_panel(
        pattern_mesh,
        positions,
        "drape_overlay",
        ("overlay_shoulder", "overlay_side"),
        base_positions=overlay_base,
        base_z_offset_m=30.0e-3,
    )

    for tie_id, seam_id, direction_xy, under_overlay in (
        # The left tie is stitched to the underlayer inside the overlay's side
        # mount.  It must leave through the open lower drape, not cross that
        # mounted edge.  The right tie is mounted directly on the overlay and
        # can fan down-right on its visible face.
        ("left_tie", "left_tie_attach", (0.25, -1.0), True),
        ("right_tie", "right_tie_attach", (0.55, -1.0), False),
    ):
        tie_mesh = _panel_mesh(pattern_mesh, tie_id)
        tie_constraints = _constraints_from_host_seams(
            pattern_mesh, positions, tie_id, (seam_id,)
        )
        anchor_global = pattern_mesh.path_vertices[(tie_id, "anchor")]
        anchor_local = anchor_global - pattern_mesh.panel_vertex_ranges[tie_id][0]
        anchor_x = tie_mesh.material_coordinates_m[anchor_local, 0]
        anchor_targets = np.asarray(
            [tie_constraints[int(index)] for index in anchor_local]
        )
        order = np.argsort(anchor_x)
        anchor_x, anchor_targets = anchor_x[order], anchor_targets[order]
        direction = np.asarray(direction_xy, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        base = np.empty((tie_mesh.vertex_count, 3), dtype=np.float64)
        tie_width = float(np.ptp(tie_mesh.material_coordinates_m[:, 0]))
        tie_length = float(np.ptp(tie_mesh.material_coordinates_m[:, 1]))
        for index, (x, y) in enumerate(tie_mesh.material_coordinates_m):
            anchor = np.asarray(
                [np.interp(x, anchor_x, anchor_targets[:, axis]) for axis in range(3)]
            )
            longitudinal = float(y)
            displacement_xy = longitudinal * direction
            if under_overlay:
                # The lower overlay opening is about 170 mm below the anchor.
                # A shallow lift keeps the strip between the two layers until
                # it exits; only then does a smooth ramp give the free length
                # visible depth.  All values are reset clearances, not textile
                # thickness or a hidden collision exception.
                exit_gate = float(
                    np.clip((longitudinal - 0.220) / 0.080, 0.0, 1.0)
                )
                exit_gate = exit_gate * exit_gate * (3.0 - 2.0 * exit_gate)
                longitudinal_arch = (
                    8.0e-3
                    * math.sin(math.pi * longitudinal / tie_length)
                    * exit_gate
                )
                clearance_arch = (
                    0.50e-3
                    * math.sin(
                        math.pi * min(longitudinal, 0.340) / 0.340
                    )
                )
                attachment_lift = 0.50e-3
            else:
                longitudinal_arch = 20.0e-3 * math.sin(
                    math.pi * longitudinal / tie_length
                )
                clearance_arch = 0.0
                attachment_lift = 4.0e-3
            attachment_arch = (
                attachment_lift
                * math.sin(math.pi * float(x) / tie_width)
                * math.exp(-float(y) / 0.040)
            )
            base[index] = anchor + np.asarray(
                (
                    float(displacement_xy[0]),
                    float(displacement_xy[1]),
                    longitudinal_arch + clearance_arch + attachment_arch,
                )
            )
        start, stop = pattern_mesh.panel_vertex_ranges[tie_id]
        positions[start:stop] = _harmonic_panel_extension(
            tie_mesh.triangles,
            tie_mesh.vertex_count,
            tie_constraints,
            base_positions=base,
        )
        constrained[tie_id] = len(tie_constraints)
        harmonic[tie_id] = tie_mesh.vertex_count - len(tie_constraints)

    constrained["front_neck_facing"], harmonic["front_neck_facing"] = _author_shaped_neck_facing(
        pattern_mesh,
        positions,
        "front_neck_facing",
        attach_seam_id="front_neck_attach",
        surface_sign=1.0,
        lift_m=10.0e-3,
    )
    constrained["back_neck_facing"], harmonic["back_neck_facing"] = _author_shaped_neck_facing(
        pattern_mesh,
        positions,
        "back_neck_facing",
        attach_seam_id="back_neck_attach",
        surface_sign=-1.0,
        lift_m=16.0e-3,
    )
    return _finish_flat_lay_strategy(
        construction,
        pattern_mesh,
        positions,
        strategy_id="asymmetric_draped_shell_flat_lay_v2",
        root_panel_id="front_under",
        body_layer_separation_m=config.body_layer_separation_m,
        constrained_counts=constrained,
        harmonic_counts=harmonic,
        notes=(
            "The front underlayer and back remain a sewn shell with unequal physical sleeve geometries.",
            "The bias-cut drape overlay is a separate attached panel, not a painted motif.",
            "Both 580 mm ties extend from their actual stitch anchors; separate curved front and back facings close into the sewn neck loop.",
        ),
    )


_STRATEGIES: Mapping[
    str,
    Callable[
        [GarmentConstruction, GarmentPatternMesh, FlatLayResetConfig],
        FlatLaySeedResult,
    ],
] = MappingProxyType(
    {
        "cotton_jersey_half_sleeve_tee": _tee_flat_lay_seed,
        "woven_button_down_shirt": _woven_shirt_flat_lay_seed,
        "denim_five_pocket_jeans": _denim_jeans_flat_lay_seed,
        "terry_cotton_bath_towel": _terry_towel_flat_lay_seed,
        "asymmetric_draped_one_sleeve_top": _asymmetric_top_flat_lay_seed,
    }
)


def build_flat_lay_seed(
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
    *,
    config: FlatLayResetConfig | None = None,
) -> FlatLaySeedResult:
    """Build a registered topology-aware flat-lay reset seed."""

    if construction.id != pattern_mesh.construction_id:
        raise ValueError("Construction and pattern mesh identifiers differ")
    try:
        strategy = _STRATEGIES[construction.id]
    except KeyError as exception:
        raise NotImplementedError(
            f"No topology-aware flat-lay strategy is registered for {construction.id!r}"
        ) from exception
    return strategy(construction, pattern_mesh, config or FlatLayResetConfig())


def supported_flat_lay_constructions() -> tuple[str, ...]:
    return tuple(_STRATEGIES)


__all__ = [
    "FlatLayResetConfig",
    "FlatLaySeedReport",
    "FlatLaySeedResult",
    "build_flat_lay_seed",
    "supported_flat_lay_constructions",
]

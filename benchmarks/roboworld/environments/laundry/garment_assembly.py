"""Sewn physical topology and initial assembly seeds for garment panels.

Newton Style3D intentionally distinguishes the physical 3-D mesh from the
2-D panel mesh: structural seam vertices are shared in physical space while
the material coordinates remain duplicated across cut panels.  This module
builds that representation from :mod:`laundry.garment_meshing`.

The generated seed is only an initial condition for a zero-gravity assembly
relaxation.  It is not accepted as a calibrated equilibrium garment merely
because its seam topology is watertight.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .garment_asset import GarmentConstruction, PathRef
from .garment_meshing import GarmentPatternMesh
from .specs import MATERIAL_PROFILES


@dataclass(frozen=True)
class RigidPanelSeedPose:
    panel_id: str
    rotation_rad: float
    translation_m: tuple[float, float]
    layer_offset_m: float
    mirror_material_x: bool = False

    @property
    def rotation_matrix(self) -> np.ndarray:
        cosine = math.cos(self.rotation_rad)
        sine = math.sin(self.rotation_rad)
        return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)

    @property
    def local_material_matrix(self) -> np.ndarray:
        """Panel-local cut orientation used only to author the assembly seed."""

        x_sign = -1.0 if self.mirror_material_x else 1.0
        return np.asarray(((x_sign, 0.0), (0.0, 1.0)), dtype=np.float64)


@dataclass(frozen=True)
class SeedAssembly:
    construction_id: str
    material_vertex_positions_m: np.ndarray
    panel_poses: Mapping[str, RigidPanelSeedPose]
    root_panel_id: str
    inter_panel_seam_rms_gap_m: float
    maximum_inter_panel_seam_rms_gap_m: float

    def __post_init__(self) -> None:
        if self.material_vertex_positions_m.ndim != 2 or self.material_vertex_positions_m.shape[1] != 3:
            raise ValueError("Seed positions must have shape (material vertices, 3)")
        if not np.all(np.isfinite(self.material_vertex_positions_m)):
            raise ValueError("Seed assembly contains non-finite positions")
        if self.root_panel_id not in self.panel_poses:
            raise ValueError("Seed root panel has no pose")
        if min(self.inter_panel_seam_rms_gap_m, self.maximum_inter_panel_seam_rms_gap_m) < 0.0:
            raise ValueError("Seed seam gaps cannot be negative")


@dataclass(frozen=True)
class AssemblyStrainMetrics:
    minimum_principal_stretch: float
    maximum_principal_stretch: float
    mean_absolute_area_strain: float
    maximum_absolute_area_strain: float
    minimum_world_triangle_area_m2: float
    degenerate_triangle_count: int


@dataclass(frozen=True)
class AssemblyRelaxationReport:
    applied: bool
    optimizer_converged: bool
    iterations: int
    initial_edge_energy: float
    final_edge_energy: float
    rms_relative_edge_strain: float
    maximum_absolute_relative_edge_strain: float
    note: str
    flat_lay_planarity_weight: float = 0.0
    flat_lay_layer_order_weight: float = 0.0
    flat_lay_layer_order_clearance_m: float = 0.0
    flat_lay_planarity_multiplier_minimum: float = 1.0
    flat_lay_planarity_multiplier_maximum: float = 1.0
    movable_vertex_count: int = 0
    relaxed_panel_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SewnGarmentTopology:
    construction_id: str
    physical_positions_m: np.ndarray
    physical_triangles: np.ndarray
    panel_material_coordinates_m: np.ndarray
    panel_triangles: np.ndarray
    material_to_physical_vertex: np.ndarray
    physical_vertex_mass_kg: np.ndarray
    physical_vertex_radius_m: np.ndarray
    physical_vertex_source_count: np.ndarray
    vertex_panel_index: np.ndarray
    triangle_panel_index: np.ndarray
    triangle_material_index: np.ndarray
    panel_ids: tuple[str, ...]
    material_profile_ids: tuple[str, ...]
    seam_physical_vertices: Mapping[str, np.ndarray]
    seam_physical_edges: Mapping[str, np.ndarray]
    boundary_edge_incidence: Mapping[tuple[int, int], int]
    seed: SeedAssembly
    relaxation: AssemblyRelaxationReport
    strain: AssemblyStrainMetrics

    def __post_init__(self) -> None:
        physical_count = len(self.physical_positions_m)
        material_count = len(self.panel_material_coordinates_m)
        triangle_count = len(self.physical_triangles)
        if self.physical_positions_m.shape != (physical_count, 3):
            raise ValueError("Physical garment positions have the wrong shape")
        if self.physical_triangles.shape != (triangle_count, 3):
            raise ValueError("Physical garment triangles have the wrong shape")
        if self.panel_triangles.shape != (triangle_count, 3):
            raise ValueError("Panel and physical triangle counts must match")
        if self.material_to_physical_vertex.shape != (material_count,):
            raise ValueError("Material-to-physical vertex map has the wrong shape")
        if self.physical_vertex_mass_kg.shape != (physical_count,):
            raise ValueError("Physical mass array has the wrong shape")
        if self.physical_vertex_radius_m.shape != (physical_count,):
            raise ValueError("Physical radius array has the wrong shape")
        if self.physical_vertex_source_count.shape != (physical_count,):
            raise ValueError("Physical source-count array has the wrong shape")
        if np.min(self.physical_triangles) < 0 or np.max(self.physical_triangles) >= physical_count:
            raise ValueError("Physical garment triangle references a missing vertex")
        if np.min(self.panel_triangles) < 0 or np.max(self.panel_triangles) >= material_count:
            raise ValueError("Panel triangle references a missing material vertex")
        if np.any(
            (self.physical_triangles[:, 0] == self.physical_triangles[:, 1])
            | (self.physical_triangles[:, 1] == self.physical_triangles[:, 2])
            | (self.physical_triangles[:, 2] == self.physical_triangles[:, 0])
        ):
            raise ValueError("Seam welding collapsed a physical garment triangle")
        if not np.all(np.isfinite(self.physical_positions_m)):
            raise ValueError("Physical garment contains non-finite positions")
        if np.min(self.physical_vertex_mass_kg) <= 0.0:
            raise ValueError("Every welded physical vertex must carry positive mass")
        if np.min(self.physical_vertex_radius_m) <= 0.0:
            raise ValueError("Every welded physical vertex must carry a positive contact radius")
        if max(self.boundary_edge_incidence.values(), default=0) > 2:
            raise ValueError(
                "Seam welding produced a non-manifold physical cloth edge with "
                "more than two incident faces"
            )

    @property
    def physical_vertex_count(self) -> int:
        return int(len(self.physical_positions_m))

    @property
    def material_vertex_count(self) -> int:
        return int(len(self.panel_material_coordinates_m))

    @property
    def triangle_count(self) -> int:
        return int(len(self.physical_triangles))

    @property
    def total_mass_kg(self) -> float:
        return float(np.sum(self.physical_vertex_mass_kg))

    @property
    def welded_material_vertex_count(self) -> int:
        return int(np.sum(self.physical_vertex_source_count > 1))


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int32)
        self.rank = np.zeros(count, dtype=np.int8)

    def find(self, value: int) -> int:
        parent = int(self.parent[value])
        if parent != value:
            self.parent[value] = self.find(parent)
        return int(self.parent[value])

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def _path_ref_points(
    construction: GarmentConstruction,
    path_ref: PathRef,
    count: int = 12,
) -> np.ndarray:
    path = construction.panel_map[path_ref.panel_id].path(path_ref.path_id)
    fractions = np.linspace(path_ref.start_fraction, path_ref.end_fraction, count)
    if path_ref.reverse:
        fractions = fractions[::-1]
    return np.asarray(
        [path.point_at_fraction(float(fraction)) for fraction in fractions],
        dtype=np.float64,
    )


def _proper_rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    u, _, vt = np.linalg.svd(
        (source - source_centroid).T @ (target - target_centroid)
    )
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - source_centroid @ rotation.T
    return rotation, translation


def _seed_local_points(
    construction: GarmentConstruction,
    panel_id: str,
    points: np.ndarray,
) -> np.ndarray:
    """Apply an explicitly authored cut-piece flip before a proper 2-D fit.

    A left/right mirrored cut is not a second material constitutive frame.  It
    is only a seed-pose choice, equivalent to turning the cut panel over before
    sewing.  Material coordinates passed to Style3D remain unchanged.
    """

    panel = construction.panel_map[panel_id]
    if not panel.seed_mirror_material_x:
        return points
    result = points.copy()
    result[:, 0] *= -1.0
    return result


def _root_panel_id(construction: GarmentConstruction) -> str:
    preferred = {
        "cotton_jersey_half_sleeve_tee": "front",
        "woven_button_down_shirt": "back",
        "denim_five_pocket_jeans": "front_left",
        "terry_cotton_bath_towel": "terry_body",
        "asymmetric_draped_one_sleeve_top": "front_under",
    }
    return preferred.get(construction.id, construction.panels[0].id)


def _layer_offset(panel_id: str, panel_index: int, panel_count: int) -> float:
    """Small topology-view separation, not a calibrated garment thickness."""

    lowered = panel_id.lower()
    if "pocket" in lowered or "overlay" in lowered or "yoke" in lowered:
        return 1.5e-3
    if "inner" in lowered or "back" in lowered:
        return -1.0e-3
    if "front" in lowered or "outer" in lowered:
        return 1.0e-3
    return (panel_index - 0.5 * (panel_count - 1)) * 0.05e-3


def build_seed_assembly(
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
) -> SeedAssembly:
    """Fit a low-gap rigid layout before welded relaxation."""

    if construction.id != pattern_mesh.construction_id:
        raise ValueError("Construction and pattern mesh identifiers differ")
    root = _root_panel_id(construction)
    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {
        root: (np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64))
    }
    while len(transforms) < len(construction.panels):
        progress = False
        for seam in construction.seams:
            if seam.side_a.panel_id in transforms and seam.side_b.panel_id not in transforms:
                placed, unplaced = seam.side_a, seam.side_b
            elif seam.side_b.panel_id in transforms and seam.side_a.panel_id not in transforms:
                placed, unplaced = seam.side_b, seam.side_a
            else:
                continue
            rotation, translation = transforms[placed.panel_id]
            target = (
                _seed_local_points(
                    construction,
                    placed.panel_id,
                    _path_ref_points(construction, placed),
                )
                @ rotation.T
                + translation
            )
            transforms[unplaced.panel_id] = _proper_rigid_fit(
                _seed_local_points(
                    construction,
                    unplaced.panel_id,
                    _path_ref_points(construction, unplaced),
                ),
                target,
            )
            progress = True
        if not progress:
            for panel in construction.panels:
                if panel.id not in transforms:
                    transforms[panel.id] = (
                        np.eye(2, dtype=np.float64),
                        np.zeros(2, dtype=np.float64),
                    )
                    progress = True
                    break
        if not progress:
            raise ValueError("Could not seed all garment panels")

    other_ids = tuple(panel.id for panel in construction.panels if panel.id != root)
    initial_variables: list[float] = []
    for panel_id in other_ids:
        rotation, translation = transforms[panel_id]
        initial_variables.extend(
            (math.atan2(rotation[1, 0], rotation[0, 0]), *translation.tolist())
        )
    initial = np.asarray(initial_variables, dtype=np.float64)

    def transformed(panel_id: str, points: np.ndarray, variables: np.ndarray) -> np.ndarray:
        points = _seed_local_points(construction, panel_id, points)
        if panel_id == root:
            return points
        offset = 3 * other_ids.index(panel_id)
        angle, tx, ty = variables[offset : offset + 3]
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        return points @ rotation.T + np.asarray((tx, ty))

    inter_panel_seams = tuple(
        seam
        for seam in construction.seams
        if seam.side_a.panel_id != seam.side_b.panel_id
    )
    optimized = initial
    if len(initial) and inter_panel_seams:
        try:
            from scipy.optimize import least_squares
        except ImportError as exception:  # pragma: no cover
            raise RuntimeError(
                "Seed assembly optimization requires SciPy; run laundry/bootstrap_newton.ps1"
            ) from exception

        def residual(variables: np.ndarray) -> np.ndarray:
            terms: list[np.ndarray] = []
            for seam in inter_panel_seams:
                side_a = transformed(
                    seam.side_a.panel_id,
                    _path_ref_points(construction, seam.side_a),
                    variables,
                )
                side_b = transformed(
                    seam.side_b.panel_id,
                    _path_ref_points(construction, seam.side_b),
                    variables,
                )
                terms.append((side_a - side_b).reshape(-1))
            # A weak seed prior removes unobservable free rotations of panels
            # connected along a perfectly straight stitch line.
            terms.append(1.0e-3 * (variables - initial))
            return np.concatenate(terms)

        result = least_squares(
            residual,
            initial,
            max_nfev=250,
            ftol=1.0e-11,
            xtol=1.0e-11,
            gtol=1.0e-11,
        )
        if not result.success:
            raise ValueError(f"Rigid seed optimization failed: {result.message}")
        optimized = result.x

    poses: dict[str, RigidPanelSeedPose] = {}
    positions = np.empty((pattern_mesh.vertex_count, 3), dtype=np.float64)
    for panel_index, panel in enumerate(construction.panels):
        if panel.id == root:
            angle, translation = 0.0, np.zeros(2, dtype=np.float64)
        else:
            offset = 3 * other_ids.index(panel.id)
            angle = float(optimized[offset])
            translation = optimized[offset + 1 : offset + 3]
        pose = RigidPanelSeedPose(
            panel_id=panel.id,
            rotation_rad=angle,
            translation_m=(float(translation[0]), float(translation[1])),
            layer_offset_m=_layer_offset(panel.id, panel_index, len(construction.panels)),
            mirror_material_x=panel.seed_mirror_material_x,
        )
        poses[panel.id] = pose
        start, stop = pattern_mesh.panel_vertex_ranges[panel.id]
        xy = (
            pattern_mesh.material_coordinates_m[start:stop]
            @ pose.local_material_matrix.T
            @ pose.rotation_matrix.T
            + np.asarray(pose.translation_m)
        )
        positions[start:stop, :2] = xy
        positions[start:stop, 2] = pose.layer_offset_m

    centre = np.average(positions, axis=0, weights=pattern_mesh.vertex_mass_kg)
    positions -= centre
    seam_rms: list[float] = []
    for stitch in pattern_mesh.stitches:
        pair_delta = positions[stitch.vertex_pairs[:, 0]] - positions[stitch.vertex_pairs[:, 1]]
        # Same-panel closures cannot be improved by a rigid panel pose but are
        # intentionally included in the reported assembly burden.
        seam_rms.append(float(np.sqrt(np.mean(np.sum(pair_delta * pair_delta, axis=1)))))
    inter_rms = [
        gap
        for gap, seam in zip(seam_rms, construction.seams)
        if seam.side_a.panel_id != seam.side_b.panel_id
    ]
    return SeedAssembly(
        construction_id=construction.id,
        material_vertex_positions_m=positions,
        panel_poses=MappingProxyType(poses),
        root_panel_id=root,
        inter_panel_seam_rms_gap_m=(
            float(np.sqrt(np.mean(np.square(inter_rms)))) if inter_rms else 0.0
        ),
        maximum_inter_panel_seam_rms_gap_m=max(inter_rms, default=0.0),
    )


def _strain_metrics(
    physical_positions: np.ndarray,
    physical_triangles: np.ndarray,
    material_coordinates: np.ndarray,
    panel_triangles: np.ndarray,
) -> AssemblyStrainMetrics:
    # Batched versions of the same per-face 3x2 deformation gradients and SVD.
    # This is an observational metric, not a new constitutive or solver model.
    world = physical_positions[physical_triangles]
    material = material_coordinates[panel_triangles]
    dm = np.stack((material[:, 1]-material[:, 0], material[:, 2]-material[:, 0]), axis=2)
    ds = np.stack((world[:, 1]-world[:, 0], world[:, 2]-world[:, 0]), axis=2)
    principal_array = np.linalg.svd(ds @ np.linalg.inv(dm), compute_uv=False)
    material_area = .5*abs(np.linalg.det(dm))
    world_areas = .5*np.linalg.norm(np.cross(ds[:, :, 0], ds[:, :, 1]), axis=1).astype(np.float64)
    area_strain_array = world_areas/material_area-1.
    return AssemblyStrainMetrics(
        minimum_principal_stretch=float(np.min(principal_array)),
        maximum_principal_stretch=float(np.max(principal_array)),
        mean_absolute_area_strain=float(np.mean(np.abs(area_strain_array))),
        maximum_absolute_area_strain=float(np.max(np.abs(area_strain_array))),
        minimum_world_triangle_area_m2=float(np.min(world_areas)),
        degenerate_triangle_count=int(np.count_nonzero(world_areas <= 1.0e-12)),
    )


def _relax_welded_positions(
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
    material_to_physical: np.ndarray,
    physical_positions: np.ndarray,
    physical_source_count: np.ndarray,
    *,
    maximum_iterations: int,
    flat_lay_planarity_weight: float = 0.0,
    flat_lay_planarity_vertex_multiplier: np.ndarray | None = None,
    flat_lay_layer_order_weight: float = 0.0,
    flat_lay_layer_order_clearance_m: float = 0.0,
    movable_vertex_mask: np.ndarray | None = None,
    relaxed_panel_ids: tuple[str, ...] = (),
) -> tuple[np.ndarray, AssemblyRelaxationReport]:
    """Reduce initial seam-closing strain while retaining the sewn topology.

    This is a deterministic geometry preconditioner, not the dynamics probe.
    It minimizes cut-panel edge-length error in 3-D, weighted by the declared
    orthotropic material family, so gathered/eased seams can buckle out of
    plane instead of forcing all mismatch into membrane stretch.
    """

    if maximum_iterations < 1:
        raise ValueError("Assembly relaxation needs at least one iteration")
    if movable_vertex_mask is None:
        movable_mask = np.ones(len(physical_positions), dtype=bool)
    else:
        movable_mask = np.asarray(movable_vertex_mask, dtype=bool)
        if movable_mask.shape != (len(physical_positions),):
            raise ValueError("Assembly movable-vertex mask has the wrong shape")
        if not np.any(movable_mask):
            raise ValueError("Assembly relaxation needs at least one movable vertex")
    if (
        not math.isfinite(flat_lay_planarity_weight)
        or flat_lay_planarity_weight < 0.0
    ):
        raise ValueError("Flat-lay planarity weight must be finite and nonnegative")
    if flat_lay_planarity_vertex_multiplier is None:
        planarity_multiplier = np.ones(len(physical_positions), dtype=np.float64)
    else:
        planarity_multiplier = np.asarray(
            flat_lay_planarity_vertex_multiplier, dtype=np.float64
        )
        if planarity_multiplier.shape != (len(physical_positions),):
            raise ValueError("Flat-lay planarity multiplier has the wrong shape")
        if (
            not np.all(np.isfinite(planarity_multiplier))
            or float(np.min(planarity_multiplier)) < 0.0
        ):
            raise ValueError(
                "Flat-lay planarity multipliers must be finite and nonnegative"
            )
    if (
        not math.isfinite(flat_lay_layer_order_weight)
        or flat_lay_layer_order_weight < 0.0
    ):
        raise ValueError("Flat-lay layer-order weight must be finite and nonnegative")
    if (
        not math.isfinite(flat_lay_layer_order_clearance_m)
        or flat_lay_layer_order_clearance_m < 0.0
    ):
        raise ValueError("Flat-lay layer-order clearance must be finite and nonnegative")
    if (flat_lay_layer_order_weight > 0.0) != (
        flat_lay_layer_order_clearance_m > 0.0
    ):
        raise ValueError(
            "Flat-lay layer-order weight and clearance must be enabled together"
        )
    material_edges: list[tuple[int, int]] = []
    physical_edges: list[tuple[int, int]] = []
    rest_lengths: list[float] = []
    stiffness_weights: list[float] = []
    seen_material_edges: set[tuple[int, int]] = set()
    panel_by_index = construction.panels
    for triangle_index, triangle in enumerate(pattern_mesh.triangles):
        panel = panel_by_index[int(pattern_mesh.triangle_panel_index[triangle_index])]
        textile = MATERIAL_PROFILES[panel.material_profile]
        grain = np.asarray(panel.grain_direction, dtype=np.float64)
        grain /= np.linalg.norm(grain)
        cross_grain = np.asarray((-grain[1], grain[0]))
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            material_edge = (min(first, second), max(first, second))
            if material_edge in seen_material_edges:
                continue
            seen_material_edges.add(material_edge)
            physical_first = int(material_to_physical[first])
            physical_second = int(material_to_physical[second])
            if physical_first == physical_second:
                continue
            rest_vector = (
                pattern_mesh.material_coordinates_m[second]
                - pattern_mesh.material_coordinates_m[first]
            )
            rest_length = float(np.linalg.norm(rest_vector))
            direction = rest_vector / rest_length
            along_grain = float(np.dot(direction, grain))
            across_grain = float(np.dot(direction, cross_grain))
            directional_weight = (
                textile.warp_membrane_n_m * along_grain**2
                + textile.weft_membrane_n_m * across_grain**2
                + 4.0
                * textile.shear_n_m
                * along_grain**2
                * across_grain**2
            )
            material_edges.append(material_edge)
            physical_edges.append((physical_first, physical_second))
            rest_lengths.append(rest_length)
            stiffness_weights.append(directional_weight)

    edge_array = np.asarray(physical_edges, dtype=np.int32)
    rest_array = np.asarray(rest_lengths, dtype=np.float64)
    weight_array = np.asarray(stiffness_weights, dtype=np.float64)
    weight_array /= float(np.median(weight_array))
    relaxation_triangles = material_to_physical[pattern_mesh.triangles]
    material_triangles = pattern_mesh.material_coordinates_m[pattern_mesh.triangles]
    material_basis = np.stack(
        (
            material_triangles[:, 1] - material_triangles[:, 0],
            material_triangles[:, 2] - material_triangles[:, 0],
        ),
        axis=2,
    )
    inverse_material_basis = np.linalg.inv(material_basis)
    initial_positions = physical_positions.copy()
    layer_midplane_z = 0.5 * (
        float(np.min(initial_positions[:, 2]))
        + float(np.max(initial_positions[:, 2]))
    )
    initial_layer_height = initial_positions[:, 2] - layer_midplane_z
    layer_sign = np.sign(initial_layer_height)
    layer_order_mask = (
        np.abs(initial_layer_height) >= flat_lay_layer_order_clearance_m
    )

    root_start, root_stop = pattern_mesh.panel_vertex_ranges[
        _root_panel_id(construction)
    ]
    candidate_material_vertices = [
        material_vertex
        for material_vertex in range(root_start, root_stop)
        if physical_source_count[int(material_to_physical[material_vertex])] == 1
    ]
    if len(candidate_material_vertices) < 3:
        raise ValueError("Root panel has too few unstitched vertices to anchor relaxation")
    candidate_uv = pattern_mesh.material_coordinates_m[candidate_material_vertices]
    selected_candidates = (
        int(np.argmin(candidate_uv[:, 0] + candidate_uv[:, 1])),
        int(np.argmax(candidate_uv[:, 0])),
        int(np.argmax(candidate_uv[:, 1])),
    )
    anchors = np.unique(
        [
            int(material_to_physical[candidate_material_vertices[index]])
            for index in selected_candidates
        ]
    )
    if len(anchors) < 3:
        # Farthest-point completion for a narrow or nearly symmetric root.
        available = np.unique(material_to_physical[candidate_material_vertices])
        while len(anchors) < 3:
            distances = np.min(
                np.linalg.norm(
                    initial_positions[available, None, :]
                    - initial_positions[anchors][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            anchors = np.unique((*anchors.tolist(), int(available[int(np.argmax(distances))])))
    free_mask = movable_mask.copy()
    free_mask[anchors] = False
    free_vertices = np.flatnonzero(free_mask)
    if len(free_vertices) == 0:
        raise ValueError("Assembly relaxation has no free vertices after anchoring")
    anchor_positions = initial_positions[anchors].copy()
    seed_regularization = 1.0e-5

    def unpack(flat_free: np.ndarray) -> np.ndarray:
        positions = initial_positions.copy()
        positions[free_vertices] = flat_free.reshape(-1, 3)
        positions[anchors] = anchor_positions
        return positions

    def objective_and_gradient(flat_free: np.ndarray) -> tuple[float, np.ndarray]:
        positions = unpack(flat_free)
        delta = positions[edge_array[:, 1]] - positions[edge_array[:, 0]]
        lengths = np.sqrt(np.sum(delta * delta, axis=1) + 1.0e-18)
        error = lengths - rest_array
        energy = 0.5 * float(np.sum(weight_array * error * error))
        coefficient = weight_array * error / lengths
        edge_gradient = coefficient[:, None] * delta
        gradient = np.zeros_like(positions)
        np.add.at(gradient, edge_array[:, 0], -edge_gradient)
        np.add.at(gradient, edge_array[:, 1], edge_gradient)

        world_triangles = positions[relaxation_triangles]
        world_basis = np.stack(
            (
                world_triangles[:, 1] - world_triangles[:, 0],
                world_triangles[:, 2] - world_triangles[:, 0],
            ),
            axis=2,
        )
        deformation = np.einsum(
            "tij,tjk->tik", world_basis, inverse_material_basis
        )
        metric = np.einsum("tki,tkj->tij", deformation, deformation)
        metric_error = metric - np.eye(2, dtype=np.float64)[None, :, :]
        # Equal per-element weighting is intentional in this initializer: it
        # prevents one tiny but highly strained triangle from being ignored
        # before the physical area-weighted solver starts.
        energy += 0.25 * float(np.sum(metric_error * metric_error))
        deformation_gradient = np.einsum(
            "tij,tjk->tik", deformation, metric_error
        )
        world_basis_gradient = np.einsum(
            "tij,tkj->tik", deformation_gradient, inverse_material_basis
        )
        triangle_gradient = np.empty_like(world_triangles)
        triangle_gradient[:, 1] = world_basis_gradient[:, :, 0]
        triangle_gradient[:, 2] = world_basis_gradient[:, :, 1]
        triangle_gradient[:, 0] = -(
            triangle_gradient[:, 1] + triangle_gradient[:, 2]
        )
        np.add.at(gradient, relaxation_triangles[:, 0], triangle_gradient[:, 0])
        np.add.at(gradient, relaxation_triangles[:, 1], triangle_gradient[:, 1])
        np.add.at(gradient, relaxation_triangles[:, 2], triangle_gradient[:, 2])

        seed_delta = positions - initial_positions
        energy += 0.5 * seed_regularization * float(np.sum(seed_delta * seed_delta))
        gradient += seed_regularization * seed_delta
        if flat_lay_planarity_weight > 0.0:
            height_delta = positions[:, 2] - initial_positions[:, 2]
            energy += 0.5 * flat_lay_planarity_weight * float(
                np.dot(planarity_multiplier, height_delta * height_delta)
            )
            gradient[:, 2] += (
                flat_lay_planarity_weight
                * planarity_multiplier
                * height_delta
            )
        if flat_lay_layer_order_weight > 0.0:
            signed_height = layer_sign * (positions[:, 2] - layer_midplane_z)
            violation = flat_lay_layer_order_clearance_m - signed_height
            active = layer_order_mask & (violation > 0.0)
            active_violation = violation[active]
            energy += 0.5 * flat_lay_layer_order_weight * float(
                np.dot(active_violation, active_violation)
            )
            gradient[active, 2] -= (
                flat_lay_layer_order_weight
                * layer_sign[active]
                * active_violation
            )
        return energy, gradient[free_vertices].reshape(-1)

    try:
        from scipy.optimize import minimize
    except ImportError as exception:  # pragma: no cover
        raise RuntimeError(
            "Welded assembly relaxation requires SciPy; run laundry/bootstrap_newton.ps1"
        ) from exception
    initial_flat = initial_positions[free_vertices].reshape(-1)
    initial_energy = objective_and_gradient(initial_flat)[0]
    result = minimize(
        objective_and_gradient,
        initial_flat,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": maximum_iterations,
            "ftol": 1.0e-15,
            "gtol": 1.0e-9,
            "maxls": 80,
        },
    )
    relaxed = unpack(result.x)
    edge_delta = relaxed[edge_array[:, 1]] - relaxed[edge_array[:, 0]]
    final_lengths = np.linalg.norm(edge_delta, axis=1)
    relative_strain = final_lengths / rest_array - 1.0
    report = AssemblyRelaxationReport(
        applied=True,
        optimizer_converged=bool(result.success),
        iterations=int(result.nit),
        initial_edge_energy=initial_energy,
        final_edge_energy=float(result.fun),
        rms_relative_edge_strain=float(np.sqrt(np.mean(relative_strain**2))),
        maximum_absolute_relative_edge_strain=float(np.max(np.abs(relative_strain))),
        note=(
            "Deterministic orthotropic edge-length preconditioner"
            + (
                " with an explicitly reported seed-height planarity penalty"
                if flat_lay_planarity_weight > 0.0
                else ""
            )
            + (
                " with a one-sided, explicitly reported flat-lay layer-order barrier"
                if flat_lay_layer_order_weight > 0.0
                else ""
            )
            + "; Newton/Style3D settling and contact probes remain required."
        ),
        flat_lay_planarity_weight=float(flat_lay_planarity_weight),
        flat_lay_layer_order_weight=float(flat_lay_layer_order_weight),
        flat_lay_layer_order_clearance_m=float(
            flat_lay_layer_order_clearance_m
        ),
        flat_lay_planarity_multiplier_minimum=float(
            np.min(planarity_multiplier)
        ),
        flat_lay_planarity_multiplier_maximum=float(
            np.max(planarity_multiplier)
        ),
        movable_vertex_count=int(len(free_vertices)),
        relaxed_panel_ids=tuple(relaxed_panel_ids),
    )
    return relaxed, report


def material_to_physical_vertex_map(
    pattern_mesh: GarmentPatternMesh,
) -> np.ndarray:
    """Return the deterministic stitch-union map used by garment welding.

    Rendering and artifact readers sometimes need the exact sewn indexing
    without re-running the zero-gravity assembly optimizer.  Keeping this map
    public prevents those readers from maintaining a second, subtly different
    seam-union implementation.
    """

    disjoint = _DisjointSet(pattern_mesh.vertex_count)
    for stitch in pattern_mesh.stitches:
        for first, second in stitch.vertex_pairs:
            disjoint.union(int(first), int(second))
    roots = np.asarray(
        [disjoint.find(index) for index in range(pattern_mesh.vertex_count)],
        dtype=np.int32,
    )
    unique_roots = np.unique(roots)
    root_to_physical = {
        int(root): index for index, root in enumerate(unique_roots.tolist())
    }
    return np.asarray(
        [root_to_physical[int(root)] for root in roots], dtype=np.int32
    )


def weld_garment_topology(
    construction: GarmentConstruction,
    pattern_mesh: GarmentPatternMesh,
    *,
    seed: SeedAssembly | None = None,
    relax: bool = True,
    relaxation_iterations: int = 1800,
    flat_lay_planarity_weight: float = 0.0,
    flat_lay_panel_planarity_multipliers: Mapping[str, float] | None = None,
    flat_lay_layer_order_weight: float = 0.0,
    flat_lay_layer_order_clearance_m: float = 0.0,
    relaxation_panel_ids: tuple[str, ...] | None = None,
) -> SewnGarmentTopology:
    """Merge stitch correspondences into Newton-style physical vertices."""

    if construction.id != pattern_mesh.construction_id:
        raise ValueError("Construction and pattern mesh identifiers differ")
    seed = seed or build_seed_assembly(construction, pattern_mesh)
    if seed.construction_id != construction.id:
        raise ValueError("Seed and construction identifiers differ")
    if len(seed.material_vertex_positions_m) != pattern_mesh.vertex_count:
        raise ValueError("Seed and pattern mesh vertex counts differ")
    flat_lay_values = (
        flat_lay_planarity_weight,
        flat_lay_layer_order_weight,
        flat_lay_layer_order_clearance_m,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in flat_lay_values):
        raise ValueError("Flat-lay relaxation controls must be finite and nonnegative")
    if (flat_lay_layer_order_weight > 0.0) != (
        flat_lay_layer_order_clearance_m > 0.0
    ):
        raise ValueError(
            "Flat-lay layer-order weight and clearance must be enabled together"
        )
    if any(value > 0.0 for value in flat_lay_values) and not (
        relax and construction.seams
    ):
        raise ValueError(
            "Flat-lay relaxation controls require enabled relaxation and sewn seams"
        )
    panel_planarity_multipliers = dict(
        flat_lay_panel_planarity_multipliers or {}
    )
    unknown_multiplier_panels = set(panel_planarity_multipliers) - set(
        pattern_mesh.panel_ids
    )
    if unknown_multiplier_panels:
        raise ValueError(
            "Flat-lay planarity multipliers reference unknown panels: "
            f"{sorted(unknown_multiplier_panels)!r}"
        )
    if panel_planarity_multipliers and flat_lay_planarity_weight <= 0.0:
        raise ValueError(
            "Panel planarity multipliers require a positive planarity weight"
        )
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in panel_planarity_multipliers.values()
    ):
        raise ValueError(
            "Panel planarity multipliers must be finite and nonnegative"
        )
    requested_relaxation_panels = (
        tuple(pattern_mesh.panel_ids)
        if relaxation_panel_ids is None
        else tuple(relaxation_panel_ids)
    )
    if not requested_relaxation_panels:
        raise ValueError("Assembly relaxation panel selection cannot be empty")
    if len(set(requested_relaxation_panels)) != len(
        requested_relaxation_panels
    ):
        raise ValueError("Assembly relaxation panel selection must be unique")
    unknown_relaxation_panels = set(requested_relaxation_panels) - set(
        pattern_mesh.panel_ids
    )
    if unknown_relaxation_panels:
        raise ValueError(
            "Assembly relaxation references unknown panels: "
            f"{sorted(unknown_relaxation_panels)!r}"
        )
    if relaxation_panel_ids is not None and not (relax and construction.seams):
        raise ValueError(
            "A panel-restricted relaxation requires enabled sewn relaxation"
        )

    material_to_physical = material_to_physical_vertex_map(pattern_mesh)
    physical_count = int(np.max(material_to_physical)) + 1
    source_count = np.zeros(physical_count, dtype=np.int32)
    np.add.at(source_count, material_to_physical, 1)
    physical_mass = np.zeros(physical_count, dtype=np.float64)
    np.add.at(physical_mass, material_to_physical, pattern_mesh.vertex_mass_kg)

    thickness_by_material_vertex = np.empty(pattern_mesh.vertex_count, dtype=np.float64)
    for panel in construction.panels:
        start, stop = pattern_mesh.panel_vertex_ranges[panel.id]
        thickness_by_material_vertex[start:stop] = panel.unloaded_thickness_m
    physical_radius = np.zeros(physical_count, dtype=np.float64)
    np.maximum.at(
        physical_radius,
        material_to_physical,
        0.5 * thickness_by_material_vertex,
    )

    weighted_positions = np.zeros((physical_count, 3), dtype=np.float64)
    np.add.at(
        weighted_positions,
        material_to_physical,
        seed.material_vertex_positions_m * pattern_mesh.vertex_mass_kg[:, None],
    )
    physical_positions = weighted_positions / physical_mass[:, None]
    physical_triangles = material_to_physical[pattern_mesh.triangles]

    material_planarity_multiplier = np.ones(
        pattern_mesh.vertex_count, dtype=np.float64
    )
    for panel_id, multiplier in panel_planarity_multipliers.items():
        start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
        material_planarity_multiplier[start:stop] = float(multiplier)
    weighted_planarity_multiplier = np.zeros(physical_count, dtype=np.float64)
    np.add.at(
        weighted_planarity_multiplier,
        material_to_physical,
        material_planarity_multiplier * pattern_mesh.vertex_mass_kg,
    )
    physical_planarity_multiplier = (
        weighted_planarity_multiplier / physical_mass
    )

    movable_vertex_mask: np.ndarray | None = None
    if relaxation_panel_ids is not None:
        selected_material_vertex = np.zeros(
            pattern_mesh.vertex_count, dtype=np.int32
        )
        for panel_id in requested_relaxation_panels:
            start, stop = pattern_mesh.panel_vertex_ranges[panel_id]
            selected_material_vertex[start:stop] = 1
        selected_source_count = np.zeros(physical_count, dtype=np.int32)
        np.add.at(
            selected_source_count,
            material_to_physical,
            selected_material_vertex,
        )
        # A stitch shared with an unselected host remains a hard Dirichlet
        # boundary.  Only vertices whose every material source belongs to the
        # selected panels can move; this prevents a sleeve-only pass from
        # silently deforming the body through welded armhole particles.
        movable_vertex_mask = selected_source_count == source_count

    if relax and construction.seams:
        physical_positions, relaxation = _relax_welded_positions(
            construction,
            pattern_mesh,
            material_to_physical,
            physical_positions,
            source_count,
            maximum_iterations=relaxation_iterations,
            flat_lay_planarity_weight=flat_lay_planarity_weight,
            flat_lay_planarity_vertex_multiplier=(
                physical_planarity_multiplier
            ),
            flat_lay_layer_order_weight=flat_lay_layer_order_weight,
            flat_lay_layer_order_clearance_m=(
                flat_lay_layer_order_clearance_m
            ),
            movable_vertex_mask=movable_vertex_mask,
            relaxed_panel_ids=requested_relaxation_panels,
        )
    else:
        relaxation = AssemblyRelaxationReport(
            applied=False,
            optimizer_converged=True,
            iterations=0,
            initial_edge_energy=0.0,
            final_edge_energy=0.0,
            rms_relative_edge_strain=0.0,
            maximum_absolute_relative_edge_strain=0.0,
            note="No sewn relaxation requested or no seams are present.",
        )

    seam_vertices: dict[str, np.ndarray] = {}
    seam_edges: dict[str, np.ndarray] = {}
    for stitch in pattern_mesh.stitches:
        first_side = material_to_physical[stitch.vertex_pairs[:, 0]]
        second_side = material_to_physical[stitch.vertex_pairs[:, 1]]
        if not np.array_equal(first_side, second_side):
            raise ValueError(f"Seam {stitch.seam_id!r} did not weld to shared vertices")
        seam_vertices[stitch.seam_id] = first_side
        seam_edges[stitch.seam_id] = np.column_stack(
            (first_side[:-1], first_side[1:])
        ).astype(np.int32)

    edge_incidence: dict[tuple[int, int], int] = {}
    for triangle in physical_triangles:
        a, b, c = (int(value) for value in triangle)
        for first, second in ((a, b), (b, c), (c, a)):
            edge = (min(first, second), max(first, second))
            edge_incidence[edge] = edge_incidence.get(edge, 0) + 1

    strain = _strain_metrics(
        physical_positions,
        physical_triangles,
        pattern_mesh.material_coordinates_m,
        pattern_mesh.triangles,
    )
    return SewnGarmentTopology(
        construction_id=construction.id,
        physical_positions_m=physical_positions,
        physical_triangles=physical_triangles,
        panel_material_coordinates_m=pattern_mesh.material_coordinates_m.copy(),
        panel_triangles=pattern_mesh.triangles.copy(),
        material_to_physical_vertex=material_to_physical,
        physical_vertex_mass_kg=physical_mass,
        physical_vertex_radius_m=physical_radius,
        physical_vertex_source_count=source_count,
        vertex_panel_index=pattern_mesh.vertex_panel_index.copy(),
        triangle_panel_index=pattern_mesh.triangle_panel_index.copy(),
        triangle_material_index=pattern_mesh.triangle_material_index.copy(),
        panel_ids=pattern_mesh.panel_ids,
        material_profile_ids=pattern_mesh.material_profile_ids,
        seam_physical_vertices=MappingProxyType(seam_vertices),
        seam_physical_edges=MappingProxyType(seam_edges),
        boundary_edge_incidence=MappingProxyType(edge_incidence),
        seed=seed,
        relaxation=relaxation,
        strain=strain,
    )

"""Newton Style3D adapter for arbitrary sewn garment constructions.

The garment compiler intentionally stores two aligned topologies:

* one welded physical mesh used for positions, contact, and bending; and
* one duplicated panel mesh used for material-space strain.

Newton 1.4 Style3D accepts this exact split through ``vertices/indices`` and
``panel_verts/panel_indices``.  This module is a narrow adapter, not a second
garment model.  It does not claim that the repository's prior-only material
numbers are calibrated measurements, and it records the Style3D features that
cannot yet express the nonlinear/hysteretic terms in :mod:`laundry.specs`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .garment_assembly import SewnGarmentTopology, weld_garment_topology
from .garment_asset import GarmentConstruction
from .garment_meshing import GarmentPatternMesh, compile_garment_pattern_mesh
from .specs import MATERIAL_PROFILES, PARAMETERS


if TYPE_CHECKING:  # pragma: no cover
    from .newton_style3d_ipc import IPCStyle3DSelfContactConfig
    from .newton_style3d_table_ccd import TablePlaneCCDConfig


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CACHE = Path(ROOT.anchor) / "garc"
WARP_CACHE = RUNTIME_CACHE / "warp"
TEMP_CACHE = RUNTIME_CACHE / "tmp"


def _configure_warp_cache() -> None:
    WARP_CACHE.mkdir(parents=True, exist_ok=True)
    TEMP_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WARP_CACHE_PATH", str(WARP_CACHE))
    os.environ["TEMP"] = str(TEMP_CACHE)
    os.environ["TMP"] = str(TEMP_CACHE)
    os.environ["TMPDIR"] = str(TEMP_CACHE)


def _require_newton() -> tuple[Any, Any]:
    _configure_warp_cache()
    try:
        import newton  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415
    except ImportError as exception:  # pragma: no cover - global test Python intentionally lacks Newton
        raise RuntimeError(
            "The sewn-garment runtime requires the repository Newton 1.4 "
            "environment; run laundry/bootstrap_newton.ps1 first."
        ) from exception
    return newton, wp


def oriented_triangle_edge_audit(
    triangles: np.ndarray,
) -> Mapping[str, int]:
    """Audit manifold incidence and opposite traversal of shared edges.

    Two consistently oriented neighboring triangles traverse their common
    edge in opposite directions. Same-direction traversal is a real surface
    orientation defect even when undirected edge incidence is still two.
    """

    faces = np.asarray(triangles, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Triangle orientation audit needs nonempty (F, 3) indices")
    if np.min(faces) < 0:
        raise ValueError("Triangle orientation audit indices cannot be negative")
    incidence: dict[tuple[int, int], list[int]] = {}
    for triangle in faces:
        a, b, c = (int(value) for value in triangle)
        if len({a, b, c}) != 3:
            raise ValueError("Triangle orientation audit found a collapsed face")
        for first, second in ((a, b), (b, c), (c, a)):
            edge = (min(first, second), max(first, second))
            direction = 1 if first < second else -1
            incidence.setdefault(edge, []).append(direction)
    boundary = 0
    shared = 0
    mismatched = 0
    nonmanifold = 0
    for directions in incidence.values():
        if len(directions) == 1:
            boundary += 1
        elif len(directions) == 2:
            shared += 1
            if directions[0] == directions[1]:
                mismatched += 1
        else:
            nonmanifold += 1
    return MappingProxyType(
        {
            "edge_count": len(incidence),
            "boundary_edge_count": boundary,
            "shared_edge_count": shared,
            "orientation_mismatch_edge_count": mismatched,
            "nonmanifold_edge_count": nonmanifold,
        }
    )


@dataclass(frozen=True)
class Style3DGarmentInputs:
    """Auditable arrays passed to Newton's official Style3D cloth helper."""

    construction_id: str
    physical_positions_m: np.ndarray
    physical_triangles: np.ndarray
    grain_aligned_panel_coordinates_m: np.ndarray
    panel_triangles: np.ndarray
    particle_mass_kg: np.ndarray
    particle_radius_m: np.ndarray
    triangle_stretch_n_m: np.ndarray
    triangle_bending_nm: np.ndarray
    triangle_membrane_damping_n_s_m: np.ndarray
    triangle_bending_damping_nm_s: np.ndarray
    triangle_panel_index: np.ndarray
    triangle_material_index: np.ndarray
    panel_ids: tuple[str, ...]
    panel_physical_winding_reversed: np.ndarray
    material_profile_ids: tuple[str, ...]
    membrane_damping_n_s_m: float
    bending_damping_nm_s: float
    unsupported_terms: tuple[str, ...]
    calibration_status: str

    def __post_init__(self) -> None:
        particle_count = len(self.physical_positions_m)
        triangle_count = len(self.physical_triangles)
        material_vertex_count = len(self.grain_aligned_panel_coordinates_m)
        if self.physical_positions_m.shape != (particle_count, 3):
            raise ValueError("Style3D physical positions must have shape (particles, 3)")
        if self.physical_triangles.shape != (triangle_count, 3):
            raise ValueError("Style3D physical triangles must have shape (triangles, 3)")
        if self.panel_triangles.shape != (triangle_count, 3):
            raise ValueError("Style3D panel and physical triangle counts must match")
        if self.grain_aligned_panel_coordinates_m.shape != (material_vertex_count, 2):
            raise ValueError("Style3D panel coordinates must have shape (material vertices, 2)")
        if self.particle_mass_kg.shape != (particle_count,):
            raise ValueError("Style3D particle mass array has the wrong shape")
        if self.particle_radius_m.shape != (particle_count,):
            raise ValueError("Style3D particle radius array has the wrong shape")
        if self.triangle_stretch_n_m.shape != (triangle_count, 3):
            raise ValueError("Style3D triangle stretch array must be (triangles, 3)")
        if self.triangle_bending_nm.shape != (triangle_count, 3):
            raise ValueError("Style3D triangle bending array must be (triangles, 3)")
        if self.triangle_membrane_damping_n_s_m.shape != (triangle_count, 3):
            raise ValueError("Style3D membrane damping array must be (triangles, 3)")
        if self.triangle_bending_damping_nm_s.shape != (triangle_count,):
            raise ValueError("Style3D bending damping array must be (triangles,)")
        if self.triangle_panel_index.shape != (triangle_count,):
            raise ValueError("Style3D triangle panel metadata has the wrong shape")
        if self.triangle_material_index.shape != (triangle_count,):
            raise ValueError("Style3D triangle material metadata has the wrong shape")
        if self.panel_physical_winding_reversed.shape != (len(self.panel_ids),):
            raise ValueError("Style3D panel winding metadata has the wrong shape")
        if self.panel_physical_winding_reversed.dtype != np.bool_:
            raise ValueError("Style3D panel winding metadata must be boolean")
        orientation = oriented_triangle_edge_audit(self.physical_triangles)
        if orientation["nonmanifold_edge_count"]:
            raise ValueError("Style3D physical mesh contains non-manifold edges")
        if orientation["orientation_mismatch_edge_count"]:
            raise ValueError(
                "Style3D physical mesh has same-direction neighboring triangles"
            )
        if min(float(np.min(self.particle_mass_kg)), float(np.min(self.particle_radius_m))) <= 0.0:
            raise ValueError("Style3D particles require positive mass and radius")
        if not all(
            np.all(np.isfinite(array))
            for array in (
                self.physical_positions_m,
                self.grain_aligned_panel_coordinates_m,
                self.particle_mass_kg,
                self.particle_radius_m,
                self.triangle_stretch_n_m,
                self.triangle_bending_nm,
                self.triangle_membrane_damping_n_s_m,
                self.triangle_bending_damping_nm_s,
            )
        ):
            raise ValueError("Style3D garment inputs contain non-finite values")
        if min(self.membrane_damping_n_s_m, self.bending_damping_nm_s) < 0.0:
            raise ValueError("Style3D damping cannot be negative")

    @property
    def particle_count(self) -> int:
        return int(len(self.physical_positions_m))

    @property
    def material_vertex_count(self) -> int:
        return int(len(self.grain_aligned_panel_coordinates_m))

    @property
    def triangle_count(self) -> int:
        return int(len(self.physical_triangles))

    @property
    def total_mass_kg(self) -> float:
        return float(np.sum(self.particle_mass_kg))


@dataclass(frozen=True)
class Style3DSolverConfig:
    """Numerical settings kept separate from textile material parameters."""

    device: str = "cuda:0"
    time_step_s: float = 1.0 / 240.0
    # The lower 10/12 example settings visibly leak free-body momentum for a
    # highly connected sewn shell.  A convergence sweep selected 20/60
    # as the audit profile; interactive profiles must be labelled lower
    # fidelity if they reduce these values.
    nonlinear_iterations: int = 20
    linear_iterations: int = 60
    gravity_m_s2: float = 0.0
    enable_self_contact: bool = False
    self_contact_radius_m: float = 3.0e-3
    self_contact_vertex_face_stiffness: float = 0.5
    self_contact_edge_edge_stiffness: float = 0.1
    self_contact_edge_face_stiffness: float = 1.0
    self_contact_topological_filter_graph_hops: int = 1
    intersection_audit_local_exclusion_graph_hops: int = 1
    intersection_audit_buffer_per_triangle: int = 256
    enable_table: bool = False
    initial_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    table_top_z_m: float = 0.0
    table_half_extent_x_m: float = 0.75
    table_half_extent_y_m: float = 0.75
    table_thickness_m: float = 0.05
    contact_stiffness_n_m: float = 5.0e4
    contact_damping_n_s_m: float = 1.0e2
    table_friction_mu: float = 0.40
    soft_contact_margin_m: float = 2.0e-3
    enable_table_normal_projection: bool = False
    enable_membrane_rate_damping: bool = False
    membrane_rate_damping_scale: float = 1.0
    enable_bending_rate_damping: bool = False
    bending_rate_damping_scale: float = 1.0
    enable_air_drag: bool = False
    air_density_kg_m3: float = 1.204
    normal_drag_coefficient: float = 1.17
    wind_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    structural_seam_bending_scale: float = 0.0
    enforce_free_rigid_frame: bool = False
    ipc_self_contact: "IPCStyle3DSelfContactConfig | None" = None
    table_plane_ccd: "TablePlaneCCDConfig | None" = None
    validate_mesh: bool = True

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("Style3D device cannot be empty")
        if not math.isfinite(self.time_step_s) or self.time_step_s <= 0.0:
            raise ValueError("Style3D time step must be finite and positive")
        if min(self.nonlinear_iterations, self.linear_iterations) < 1:
            raise ValueError("Style3D iteration counts must be positive")
        if not math.isfinite(self.gravity_m_s2) or self.gravity_m_s2 < 0.0:
            raise ValueError("Style3D gravity magnitude must be finite and nonnegative")
        if not math.isfinite(self.self_contact_radius_m) or self.self_contact_radius_m <= 0.0:
            raise ValueError("Style3D self-contact radius must be finite and positive")
        if min(
            self.self_contact_vertex_face_stiffness,
            self.self_contact_edge_edge_stiffness,
            self.self_contact_edge_face_stiffness,
        ) < 0.0:
            raise ValueError("Style3D self-contact stiffness factors cannot be negative")
        if self.self_contact_topological_filter_graph_hops < 0:
            raise ValueError("Style3D self-contact filter graph hops cannot be negative")
        if self.intersection_audit_local_exclusion_graph_hops < 0:
            raise ValueError("Style3D intersection-audit graph hops cannot be negative")
        if self.intersection_audit_buffer_per_triangle < 1:
            raise ValueError("Style3D intersection-audit buffer size must be positive")
        if len(self.initial_translation_m) != 3 or not all(
            math.isfinite(value) for value in self.initial_translation_m
        ):
            raise ValueError("Style3D initial translation must be a finite 3-vector")
        if min(
            self.table_half_extent_x_m,
            self.table_half_extent_y_m,
            self.table_thickness_m,
            self.contact_stiffness_n_m,
            self.contact_damping_n_s_m,
            self.soft_contact_margin_m,
        ) <= 0.0:
            raise ValueError("Style3D table/contact dimensions and coefficients must be positive")
        if self.table_friction_mu < 0.0 or not math.isfinite(self.table_friction_mu):
            raise ValueError("Style3D table friction must be finite and nonnegative")
        if not math.isfinite(self.membrane_rate_damping_scale) or self.membrane_rate_damping_scale < 0.0:
            raise ValueError("Style3D material damping scale must be finite and nonnegative")
        if not math.isfinite(self.bending_rate_damping_scale) or self.bending_rate_damping_scale < 0.0:
            raise ValueError("Style3D bending damping scale must be finite and nonnegative")
        if (
            not math.isfinite(self.air_density_kg_m3)
            or not math.isfinite(self.normal_drag_coefficient)
            or min(self.air_density_kg_m3, self.normal_drag_coefficient) < 0.0
        ):
            raise ValueError("Style3D air density and drag coefficient must be finite and nonnegative")
        if len(self.wind_velocity_m_s) != 3 or not all(
            math.isfinite(value) for value in self.wind_velocity_m_s
        ):
            raise ValueError("Style3D wind velocity must be a finite 3-vector")
        if (
            not math.isfinite(self.structural_seam_bending_scale)
            or self.structural_seam_bending_scale < 0.0
        ):
            raise ValueError("Style3D seam bending scale must be finite and nonnegative")
        if self.enforce_free_rigid_frame and (
            self.gravity_m_s2 != 0.0 or self.enable_table
        ):
            raise ValueError(
                "The free rigid-frame gauge is valid only without gravity/external contact; "
                "external impulse accounting is required before using it in a task solve"
            )
        if self.enable_table_normal_projection and not self.enable_table:
            raise ValueError("Table normal projection requires an enabled table")
        if self.table_plane_ccd is not None and not self.enable_table:
            raise ValueError("Table-plane CCD requires an enabled table")
        if self.table_plane_ccd is not None and self.enable_table_normal_projection:
            raise ValueError(
                "Choose the swept table-plane response or the legacy residual "
                "projection, not both"
            )


def _grain_aligned_panel_coordinates(
    construction: GarmentConstruction,
    topology: SewnGarmentTopology,
) -> np.ndarray:
    """Rotate each 2-D cut panel so ``u=weft`` and ``v=warp/grain``."""

    result = topology.panel_material_coordinates_m.astype(np.float64, copy=True)
    for panel_index, panel_id in enumerate(topology.panel_ids):
        panel = construction.panel_map[panel_id]
        grain = np.asarray(panel.grain_direction, dtype=np.float64)
        grain /= np.linalg.norm(grain)
        # [weft, warp] has determinant +1, preserving triangle winding.
        weft = np.asarray((grain[1], -grain[0]), dtype=np.float64)
        rotation = np.column_stack((weft, grain))
        mask = topology.vertex_panel_index == panel_index
        origin = result[np.flatnonzero(mask)[0]].copy()
        result[mask] = (result[mask] - origin) @ rotation
    return result


def build_style3d_inputs(
    construction: GarmentConstruction,
    topology: SewnGarmentTopology,
) -> Style3DGarmentInputs:
    """Translate a validated sewn topology into engine-facing SI arrays."""

    if construction.id != topology.construction_id:
        raise ValueError("Construction and sewn topology identifiers differ")
    profiles = tuple(MATERIAL_PROFILES[name] for name in topology.material_profile_ids)
    triangle_stretch = np.asarray(
        [
            (
                profiles[int(material_index)].weft_membrane_n_m,
                profiles[int(material_index)].warp_membrane_n_m,
                profiles[int(material_index)].shear_n_m,
            )
            for material_index in topology.triangle_material_index
        ],
        dtype=np.float64,
    )
    triangle_bending = np.asarray(
        [
            (
                profiles[int(material_index)].weft_bend_nm,
                profiles[int(material_index)].warp_bend_nm,
                profiles[int(material_index)].bias_bend_nm,
            )
            for material_index in topology.triangle_material_index
        ],
        dtype=np.float64,
    )
    triangle_membrane_damping = np.asarray(
        [
            (profiles[int(index)].membrane_damping_n_s_m,) * 3
            for index in topology.triangle_material_index
        ],
        dtype=np.float64,
    )
    triangle_bending_damping = np.asarray(
        [profiles[int(index)].bend_damping_nm_s for index in topology.triangle_material_index],
        dtype=np.float64,
    )
    panel_coordinates = _grain_aligned_panel_coordinates(construction, topology)
    physical_triangles = topology.physical_triangles.astype(np.int32, copy=True)
    panel_triangles = topology.panel_triangles.astype(np.int32, copy=True)
    panel_winding_reversed = np.asarray(
        [
            construction.panel_map[panel_id].physical_winding_reversed
            for panel_id in topology.panel_ids
        ],
        dtype=np.bool_,
    )
    # Newton aligns each physical triangle corner with the corresponding
    # material-space corner.  A turned-over cut panel therefore needs both
    # index triplets reversed; reflecting its local material u coordinate
    # keeps the panel-space signed area positive without changing strain.
    for panel_index, reversed_winding in enumerate(panel_winding_reversed):
        if not reversed_winding:
            continue
        vertex_mask = topology.vertex_panel_index == panel_index
        panel_coordinates[vertex_mask, 0] *= -1.0
        triangle_mask = topology.triangle_panel_index == panel_index
        physical_triangles[triangle_mask, 1:3] = physical_triangles[
            triangle_mask, 2:0:-1
        ]
        panel_triangles[triangle_mask, 1:3] = panel_triangles[
            triangle_mask, 2:0:-1
        ]
    p = panel_coordinates[panel_triangles[:, 0]]
    q = panel_coordinates[panel_triangles[:, 1]]
    r = panel_coordinates[panel_triangles[:, 2]]
    signed_double_area = (
        (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1])
        - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])
    )
    if np.min(signed_double_area) <= 0.0:
        raise ValueError("Grain alignment inverted or collapsed a panel triangle")
    triangle_area = 0.5 * signed_double_area
    total_area = float(np.sum(triangle_area))
    membrane_damping = float(
        np.sum(
            triangle_area
            * np.asarray(
                [profiles[int(index)].membrane_damping_n_s_m for index in topology.triangle_material_index]
            )
        )
        / total_area
    )
    bending_damping = float(
        np.sum(
            triangle_area
            * np.asarray(
                [profiles[int(index)].bend_damping_nm_s for index in topology.triangle_material_index]
            )
        )
        / total_area
    )
    return Style3DGarmentInputs(
        construction_id=construction.id,
        physical_positions_m=topology.physical_positions_m.astype(np.float64, copy=True),
        physical_triangles=physical_triangles,
        grain_aligned_panel_coordinates_m=panel_coordinates,
        panel_triangles=panel_triangles,
        particle_mass_kg=topology.physical_vertex_mass_kg.astype(np.float64, copy=True),
        particle_radius_m=topology.physical_vertex_radius_m.astype(np.float64, copy=True),
        triangle_stretch_n_m=triangle_stretch,
        triangle_bending_nm=triangle_bending,
        triangle_membrane_damping_n_s_m=triangle_membrane_damping,
        triangle_bending_damping_nm_s=triangle_bending_damping,
        triangle_panel_index=topology.triangle_panel_index.astype(np.int32, copy=True),
        triangle_material_index=topology.triangle_material_index.astype(np.int32, copy=True),
        panel_ids=topology.panel_ids,
        panel_physical_winding_reversed=panel_winding_reversed,
        material_profile_ids=topology.material_profile_ids,
        membrane_damping_n_s_m=membrane_damping,
        bending_damping_nm_s=bending_damping,
        unsupported_terms=(
            "warp/weft tension stiffening is nonlinear in the authored profiles but Style3D 1.4 uses constant triangle stiffness",
            "cross-coupling ratio is not exposed by the Style3D 1.4 cloth helper",
            "pressure-dependent thickness/compression is not exposed by the Style3D 1.4 cloth helper",
            "measured per-seam rest angle and moment-curvature are unavailable; the explicit structural-seam bending scale defaults to a free hinge instead of silently treating a seam as continuous fabric",
            "Style3D 1.4 does not consume its stored damping fields; the adapter can explicitly apply objective membrane-rate and dihedral-rate damping",
            "quadratic normal air drag omits textile permeability, surrounding-flow coupling, wakes, and lift",
        ),
        calibration_status=(
            "construction-complete; engine adapter verified structurally; "
            "material and seam coefficients remain prior-only"
        ),
    )


def _structural_seam_edge_mask(
    newton: Any,
    inputs: Style3DGarmentInputs,
    topology: SewnGarmentTopology,
) -> np.ndarray:
    """Match Newton's physical hinge edges to authored structural stitches."""

    adjacency = newton.utils.MeshAdjacency(inputs.physical_triangles)
    edge_indices = np.asarray(adjacency.edge_indices, dtype=np.int64)
    seam_keys = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edges in topology.seam_physical_edges.values()
        for edge in edges
    }
    return np.asarray(
        [
            tuple(sorted((int(edge[2]), int(edge[3])))) in seam_keys
            for edge in edge_indices
        ],
        dtype=np.bool_,
    )


def _edge_bending_values(
    newton: Any,
    inputs: Style3DGarmentInputs,
    topology: SewnGarmentTopology,
    *,
    structural_seam_scale: float,
) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    adjacency = newton.utils.MeshAdjacency(inputs.physical_triangles)
    adjacent_triangles = np.asarray(adjacency.edge_tri_indices, dtype=np.int64)
    if adjacent_triangles.ndim != 2 or adjacent_triangles.shape[1] != 2:
        raise ValueError("Newton returned an unexpected cloth-edge adjacency layout")
    if np.min(adjacent_triangles[:, 0]) < 0:
        raise ValueError("Newton returned a cloth edge without a primary incident face")
    first = inputs.triangle_bending_nm[adjacent_triangles[:, 0]]
    second = first.copy()
    interior = adjacent_triangles[:, 1] >= 0
    second[interior] = inputs.triangle_bending_nm[adjacent_triangles[interior, 1]]
    # A seam hinge couples two adjacent panels.  The geometric mean is a
    # symmetric positive prior that cannot exceed the stiffer side; measured
    # seam moment-curvature data must eventually replace it.
    blended = np.sqrt(first * second)
    seam_mask = _structural_seam_edge_mask(newton, inputs, topology)
    blended[seam_mask] *= structural_seam_scale
    return [tuple(float(value) for value in row) for row in blended], seam_mask


def _edge_bending_damping_values(
    newton: Any,
    inputs: Style3DGarmentInputs,
) -> np.ndarray:
    """Blend adjacent material hinge damping without inventing a stiff side."""

    adjacency = newton.utils.MeshAdjacency(inputs.physical_triangles)
    adjacent = np.asarray(adjacency.edge_tri_indices, dtype=np.int64)
    first = inputs.triangle_bending_damping_nm_s[adjacent[:, 0]]
    second = first.copy()
    interior = adjacent[:, 1] >= 0
    second[interior] = inputs.triangle_bending_damping_nm_s[
        adjacent[interior, 1]
    ]
    return np.sqrt(first * second)


def style3d_projective_neighbor_counts(physical_triangles: np.ndarray) -> np.ndarray:
    """Reproduce Style3D's stretch+bending sparse-matrix graph in Python."""

    triangles = np.asarray(physical_triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or np.min(triangles) < 0:
        raise ValueError("Physical triangles must be nonnegative with shape (count, 3)")
    vertex_count = int(np.max(triangles)) + 1
    neighbors = [set() for _ in range(vertex_count)]
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, triangle in enumerate(triangles):
        a, b, c = (int(value) for value in triangle)
        for vertex, others in ((a, (b, c)), (b, (a, c)), (c, (a, b))):
            neighbors[vertex].update(others)
        for first, second in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((first, second)))
            edge_faces.setdefault(edge, []).append(face_id)
    for (first, second), face_ids in edge_faces.items():
        if len(face_ids) != 2:
            continue
        opposite = []
        for face_id in face_ids:
            opposite.append(
                next(
                    int(vertex)
                    for vertex in triangles[face_id]
                    if int(vertex) not in (first, second)
                )
            )
        stencil = (opposite[0], opposite[1], first, second)
        for vertex in stencil:
            neighbors[vertex].update(other for other in stencil if other != vertex)
    return np.asarray([len(values) for values in neighbors], dtype=np.int32)


class Style3DGarmentSimulation:
    """Minimal official-Newton runtime used by garment physics probes."""

    def __init__(
        self,
        construction: GarmentConstruction,
        pattern_mesh: GarmentPatternMesh,
        topology: SewnGarmentTopology,
        *,
        config: Style3DSolverConfig | None = None,
    ) -> None:
        self.construction = construction
        self.pattern_mesh = pattern_mesh
        self.topology = topology
        self.config = config or Style3DSolverConfig()
        self.inputs = build_style3d_inputs(construction, topology)
        self.newton, self.wp = _require_newton()
        self._build()

    @classmethod
    def from_construction(
        cls,
        construction: GarmentConstruction,
        *,
        target_edge_length_m: float = 0.040,
        assembly_relaxation_iterations: int = 1200,
        config: Style3DSolverConfig | None = None,
    ) -> "Style3DGarmentSimulation":
        pattern_mesh = compile_garment_pattern_mesh(
            construction,
            target_edge_length_m=target_edge_length_m,
        )
        topology = weld_garment_topology(
            construction,
            pattern_mesh,
            relax=True,
            relaxation_iterations=assembly_relaxation_iterations,
        )
        return cls(construction, pattern_mesh, topology, config=config)

    def _build(self) -> None:
        newton, wp = self.newton, self.wp
        projective_neighbors = style3d_projective_neighbor_counts(
            self.inputs.physical_triangles
        )
        if int(np.max(projective_neighbors)) > 32:
            vertex = int(np.argmax(projective_neighbors))
            raise ValueError(
                f"Style3D projective matrix needs {int(projective_neighbors[vertex])} "
                f"neighbors at physical vertex {vertex}, exceeding Newton 1.4's hard "
                "limit of 32"
            )
        self.projective_neighbor_counts = projective_neighbors
        # Scalar gravity is supported by both retained 1.4 and target 1.5.
        # 1.4's builder cannot consume the vector form introduced in 1.5.
        builder = newton.ModelBuilder(gravity=-self.config.gravity_m_s2)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        if self.config.enable_table:
            table_cfg = newton.ModelBuilder.ShapeConfig(
                ke=self.config.contact_stiffness_n_m,
                kd=self.config.contact_damping_n_s_m,
                mu=self.config.table_friction_mu,
                margin=0.0,
            )
            self.table_shape = builder.add_shape_box(
                -1,
                xform=wp.transform(
                    wp.vec3(
                        0.0,
                        0.0,
                        self.config.table_top_z_m - self.config.table_thickness_m,
                    ),
                    wp.quat_identity(),
                ),
                hx=self.config.table_half_extent_x_m,
                hy=self.config.table_half_extent_y_m,
                hz=self.config.table_thickness_m,
                cfg=table_cfg,
                label="laundry_style3d_table",
            )
        edge_bending, structural_seam_edge_mask = _edge_bending_values(
            newton,
            self.inputs,
            self.topology,
            structural_seam_scale=self.config.structural_seam_bending_scale,
        )
        edge_bending_damping = _edge_bending_damping_values(newton, self.inputs)
        self.structural_seam_edge_mask = structural_seam_edge_mask
        newton.solvers.style3d.add_cloth_mesh(
            builder,
            pos=wp.vec3(*self.config.initial_translation_m),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=self.inputs.physical_positions_m.tolist(),
            indices=self.inputs.physical_triangles.reshape(-1).tolist(),
            density=1.0,
            panel_verts=self.inputs.grain_aligned_panel_coordinates_m.tolist(),
            panel_indices=self.inputs.panel_triangles.reshape(-1).tolist(),
            tri_aniso_ke=[tuple(row) for row in self.inputs.triangle_stretch_n_m.tolist()],
            edge_aniso_ke=edge_bending,
            tri_ka=0.0,
            tri_kd=self.inputs.membrane_damping_n_s_m,
            tri_drag=0.0,
            tri_lift=0.0,
            edge_kd=self.inputs.bending_damping_nm_s,
            add_springs=False,
            particle_radius=float(np.median(self.inputs.particle_radius_m)),
            validate_mesh=self.config.validate_mesh,
            label=self.construction.id,
        )
        if len(builder.particle_mass) != self.inputs.particle_count:
            raise ValueError("Style3D helper did not preserve the physical particle count")
        if len(builder.tri_indices) != self.inputs.triangle_count:
            raise ValueError("Style3D helper discarded a panel triangle")
        builder.particle_mass[:] = self.inputs.particle_mass_kg.tolist()
        builder.particle_radius[:] = self.inputs.particle_radius_m.tolist()
        cloth_particle_count = len(builder.particle_mass)
        cloth_triangle_count = len(builder.tri_indices)
        self._extend_builder(builder)
        if len(builder.particle_mass) != cloth_particle_count:
            raise ValueError("A Style3D world extension cannot add untracked particles")
        if len(builder.tri_indices) != cloth_triangle_count:
            raise ValueError("A Style3D world extension cannot alter cloth triangles")
        self.model = builder.finalize(device=self.config.device)
        self._model_mass_kg = self.model.particle_mass.numpy().astype(np.float64)
        if self.config.enable_membrane_rate_damping:
            scale = self.config.membrane_rate_damping_scale
            self.triangle_damping_wp = wp.array(
                self.inputs.triangle_membrane_damping_n_s_m * scale,
                dtype=wp.vec3,
                device=self.config.device,
            )
        else:
            self.triangle_damping_wp = None
        if self.config.enable_bending_rate_damping:
            self.edge_bending_damping_wp = wp.array(
                edge_bending_damping * self.config.bending_rate_damping_scale,
                dtype=float,
                device=self.config.device,
            )
        else:
            self.edge_bending_damping_wp = None
        self.model.soft_contact_ke = self.config.contact_stiffness_n_m
        self.model.soft_contact_kd = self.config.contact_damping_n_s_m
        self.model.soft_contact_mu = self.config.table_friction_mu
        self.solver = newton.solvers.SolverStyle3D(
            model=self.model,
            iterations=self.config.nonlinear_iterations,
            linear_iterations=self.config.linear_iterations,
        )
        has_rigid_shapes = self.model.shape_count > 0
        if self.config.enable_self_contact:
            if self.config.self_contact_topological_filter_graph_hops > 0:
                from .newton_style3d_topology_contact import (  # noqa: PLC0415
                    TopologicallyFilteredStyle3DCollision,
                )

                self.solver.collision = TopologicallyFilteredStyle3DCollision(
                    self.model,
                    self.inputs.physical_triangles,
                    graph_hops=(
                        self.config.self_contact_topological_filter_graph_hops
                    ),
                )
            self.solver.collision.radius = self.config.self_contact_radius_m
            self.solver.collision.stiff_vf = self.config.self_contact_vertex_face_stiffness
            self.solver.collision.stiff_ee = self.config.self_contact_edge_edge_stiffness
            self.solver.collision.stiff_ef = self.config.self_contact_edge_face_stiffness
        elif has_rigid_shapes:
            self.solver.collision.stiff_vf = 0.0
            self.solver.collision.stiff_ee = 0.0
            self.solver.collision.stiff_ef = 0.0
        else:
            self.solver.collision = None
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self._initialize_state()
        if has_rigid_shapes:
            self.pipeline = newton.CollisionPipeline(
                self.model,
                broad_phase="nxn",
                soft_contact_margin=self.config.soft_contact_margin_m,
                enable_rigid_soft_full_surface_contact=False,
            )
            self.contacts = self.pipeline.contacts()
            self.table_projection_diagnostics = (
                wp.zeros(5, dtype=float, device=self.config.device)
                if self.config.enable_table
                else None
            )
        else:
            self.pipeline = None
            self.contacts = self.model.contacts()
            self.table_projection_diagnostics = None
        self.step_count = 0
        from .self_intersection import NewtonTriangleIntersectionAuditor  # noqa: PLC0415

        self._self_intersection_auditor = NewtonTriangleIntersectionAuditor(
            self.model,
            self.inputs.physical_triangles,
            local_exclusion_graph_hops=(
                self.config.intersection_audit_local_exclusion_graph_hops
            ),
            buffer_per_triangle=self.config.intersection_audit_buffer_per_triangle,
        )
        self._ipc_self_contact_controller = None
        if self.config.ipc_self_contact is not None:
            from .newton_style3d_ipc import (  # noqa: PLC0415
                IPCStyle3DSelfContactController,
            )

            self._ipc_self_contact_controller = IPCStyle3DSelfContactController(
                self,
                self.config.ipc_self_contact,
            )
        self._table_plane_ccd_controller = None
        if self.config.table_plane_ccd is not None:
            from .newton_style3d_table_ccd import (  # noqa: PLC0415
                TablePlaneCCDController,
            )

            self._table_plane_ccd_controller = TablePlaneCCDController(
                self,
                self.config.table_plane_ccd,
            )
        self.initial_center_of_mass_m = self.center_of_mass_m()
        self._after_build()

    def _extend_builder(self, builder) -> None:
        """Hook for rigid tools authored before model finalization.

        Extensions may add bodies, joints, and collision shapes, but not cloth
        particles or triangles.  This keeps the material/topology audit exact.
        """

    def _after_build(self) -> None:
        """Hook for extension buffers that require a finalized model/state."""

    def _initialize_state(self) -> None:
        """Install a selected reset after rest data, before integrity guards.

        The default baseline uses model initial positions. Extensions may load
        a separately audited reset without changing material/rest quantities.
        """

    def _prepare_external_shapes(self, dt: float) -> None:
        """Advance externally controlled rigid shapes before collision."""

    def step(self, count: int = 1) -> None:
        if count < 1:
            raise ValueError("Style3D step count must be positive")
        for _ in range(count):
            previous_positions = (
                self.positions_m().astype(np.float64)
                if self.config.enforce_free_rigid_frame
                else None
            )
            self.state_0.clear_forces()
            table_ccd_origin = (
                self._table_plane_ccd_controller.begin_step(self)
                if self._table_plane_ccd_controller is not None
                else None
            )
            ipc_origin = (
                self._ipc_self_contact_controller.begin_step(self)
                if self._ipc_self_contact_controller is not None
                else None
            )
            self._prepare_external_shapes(self.config.time_step_s)
            self._apply_membrane_rate_damping()
            self._apply_bending_rate_damping()
            self._apply_air_drag()
            if self.pipeline is not None:
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.config.time_step_s,
            )
            self._project_residual_table_contact()
            if previous_positions is not None:
                self._project_free_rigid_frame(previous_positions)
            if self._table_plane_ccd_controller is not None:
                self._table_plane_ccd_controller.finish_step(
                    self,
                    table_ccd_origin,
                )
            if self._ipc_self_contact_controller is not None:
                self._ipc_self_contact_controller.finish_step(
                    self,
                    ipc_origin,
                )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.step_count += 1

    def _project_residual_table_contact(self) -> None:
        if not self.config.enable_table_normal_projection:
            return
        from .newton_style3d_contact_kernels import (  # noqa: PLC0415
            project_residual_table_normal_contact,
        )

        self.wp.launch(
            project_residual_table_normal_contact,
            dim=self.model.particle_count,
            inputs=[
                self.model.particle_mass,
                self.model.particle_radius,
                self.config.table_top_z_m,
                self.config.table_half_extent_x_m,
                self.config.table_half_extent_y_m,
            ],
            outputs=[
                self.state_1.particle_q,
                self.state_1.particle_qd,
                self.table_projection_diagnostics,
            ],
            device=self.config.device,
        )

    def table_projection_summary(self) -> Mapping[str, float | int | bool]:
        if self.table_projection_diagnostics is None:
            return {"enabled": False}
        values = self.table_projection_diagnostics.numpy()
        return {
            "enabled": self.config.enable_table_normal_projection,
            "cumulative_normal_impulse_kg_m_s": float(values[0]),
            "cumulative_position_correction_m": float(values[1]),
            "cumulative_removed_normal_energy_j": float(values[2]),
            "corrected_particle_events": int(round(float(values[3]))),
            "maximum_single_position_correction_m": float(values[4]),
        }

    def table_plane_ccd_summary(self) -> Mapping[str, object]:
        if self._table_plane_ccd_controller is None:
            return {"enabled": False}
        return self._table_plane_ccd_controller.summary()

    def _apply_membrane_rate_damping(self) -> None:
        if not self.config.enable_membrane_rate_damping:
            return
        from .newton_style3d_damping_kernels import (  # noqa: PLC0415
            accumulate_anisotropic_membrane_damping,
        )

        self.wp.launch(
            accumulate_anisotropic_membrane_damping,
            dim=self.model.tri_count,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.model.tri_areas,
                self.model.tri_poses,
                self.model.tri_indices,
                self.triangle_damping_wp,
            ],
            outputs=[self.state_0.particle_f],
            device=self.config.device,
        )

    def _apply_bending_rate_damping(self) -> None:
        if not self.config.enable_bending_rate_damping:
            return
        from .newton_style3d_damping_kernels import (  # noqa: PLC0415
            accumulate_dihedral_rate_damping,
        )

        self.wp.launch(
            accumulate_dihedral_rate_damping,
            dim=self.model.edge_count,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.model.edge_indices,
                self.edge_bending_damping_wp,
            ],
            outputs=[self.state_0.particle_f],
            device=self.config.device,
        )

    def _apply_air_drag(self) -> None:
        if not self.config.enable_air_drag:
            return
        from .newton_style3d_damping_kernels import (  # noqa: PLC0415
            accumulate_quadratic_normal_air_drag,
        )

        self.wp.launch(
            accumulate_quadratic_normal_air_drag,
            dim=self.model.tri_count,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.model.tri_indices,
                self.config.air_density_kg_m3,
                self.config.normal_drag_coefficient,
                self.wp.vec3(*self.config.wind_velocity_m_s),
            ],
            outputs=[self.state_0.particle_f],
            device=self.config.device,
        )

    def _project_free_rigid_frame(self, previous_positions: np.ndarray) -> None:
        """Remove solver-created rigid modes during force-free equilibration.

        This is a gauge projection, not a task force or pose controller.  The
        post-solve shape is preserved up to one best-fit rigid transform.  A
        mass-minimum velocity correction then enforces the exact zero external
        linear/angular impulse of this restricted probe configuration.
        """

        positions = self.state_1.particle_q.numpy().astype(np.float64)
        mass = self._model_mass_kg
        total_mass = float(np.sum(mass))
        previous_center = np.sum(previous_positions * mass[:, None], axis=0) / total_mass
        current_center = np.sum(positions * mass[:, None], axis=0) / total_mass
        previous_centered = previous_positions - previous_center
        current_centered = positions - current_center
        covariance = (current_centered * mass[:, None]).T @ previous_centered
        left, _, right_transpose = np.linalg.svd(covariance)
        rotation = left @ right_transpose
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right_transpose
        corrected_positions = current_centered @ rotation + previous_center
        velocity = (corrected_positions - previous_positions) / self.config.time_step_s
        velocity -= np.sum(velocity * mass[:, None], axis=0) / total_mass

        relative = corrected_positions - previous_center
        inertia = np.zeros((3, 3), dtype=np.float64)
        identity = np.eye(3, dtype=np.float64)
        for weight, offset in zip(mass, relative):
            inertia += weight * (
                float(np.dot(offset, offset)) * identity - np.outer(offset, offset)
            )
        angular_momentum = np.sum(
            np.cross(relative, velocity) * mass[:, None],
            axis=0,
        )
        angular_velocity = np.linalg.solve(inertia, angular_momentum)
        velocity -= np.cross(angular_velocity, relative)

        self.state_1.particle_q.assign(corrected_positions.astype(np.float32))
        self.state_1.particle_qd.assign(velocity.astype(np.float32))

    def read_contact_arrays(self, **arrays):
        """Host contact inputs; workcell adapters may batch equivalent copies."""
        return {name:array.numpy() for name,array in arrays.items()}

    def positions_m(self) -> np.ndarray:
        return self.state_0.particle_q.numpy()

    def velocities_m_s(self) -> np.ndarray:
        return self.state_0.particle_qd.numpy()

    def self_intersection_pairs(self) -> np.ndarray:
        """Return every robustly confirmed non-adjacent surface intersection.

        IPC natively excludes primitives that share mesh adjacency.  The
        additional graph-local split remains useful diagnostic context, but it
        is not permission for a sewn shell to cross itself.
        """

        return self._self_intersection_auditor.confirmed_pairs(
            self.state_0.particle_q
        )

    def self_intersection_audit_counts(self) -> dict[str, int]:
        """Partition exact triangle intersections into nonlocal/local pairs."""

        return self._self_intersection_auditor.counts(
            self.state_0.particle_q
        ).as_dict()

    def self_intersection_count(self) -> int:
        """Return the all-confirmed count consumed by integrity success gates."""

        return int(self.self_intersection_pairs().shape[0])

    def self_contact_topological_filter_summary(self) -> Mapping[str, int | bool]:
        collision = self.solver.collision
        if collision is None or not hasattr(collision, "filter_summary"):
            return {"enabled": False}
        return {
            "enabled": True,
            "graph_hops": self.config.self_contact_topological_filter_graph_hops,
            **collision.filter_summary(),
        }

    def ipc_self_contact_summary(self) -> Mapping[str, object]:
        if self._ipc_self_contact_controller is None:
            return {"enabled": False}
        return self._ipc_self_contact_controller.summary()

    def author_state(
        self,
        positions_m: np.ndarray,
        velocities_m_s: np.ndarray | None = None,
        *,
        reset_clock: bool = True,
    ) -> None:
        """Install an audited initial state without changing topology/rest data."""

        positions = np.asarray(positions_m, dtype=np.float32)
        if positions.shape != (self.inputs.particle_count, 3) or not np.all(np.isfinite(positions)):
            raise ValueError("Authored Style3D positions must be finite (particle_count, 3)")
        if velocities_m_s is None:
            velocities = np.zeros_like(positions)
        else:
            velocities = np.asarray(velocities_m_s, dtype=np.float32)
            if velocities.shape != positions.shape or not np.all(np.isfinite(velocities)):
                raise ValueError("Authored Style3D velocities must match the finite position array")
        if self._ipc_self_contact_controller is not None:
            self._ipc_self_contact_controller.validate_positions(
                positions,
                context="authored state",
            )
        if self._table_plane_ccd_controller is not None:
            self._table_plane_ccd_controller.validate_positions(
                positions,
                self,
                context="authored state",
            )
        for state in (self.state_0, self.state_1):
            state.particle_q.assign(positions)
            state.particle_qd.assign(velocities)
            state.clear_forces()
        if self.solver.collision is not None:
            self.solver.rebuild_bvh(self.state_0)
        if reset_clock:
            self.step_count = 0
        self.initial_center_of_mass_m = self.center_of_mass_m()

    def set_self_contact_parameters(
        self,
        *,
        radius_m: float,
        vertex_face_stiffness: float,
        edge_edge_stiffness: float,
        edge_face_stiffness: float,
    ) -> None:
        """Update explicit Style3D continuation parameters and rebuild BVHs."""

        if not self.config.enable_self_contact or self.solver.collision is None:
            raise ValueError("Cannot tune self-contact when it is disabled")
        values = (
            radius_m,
            vertex_face_stiffness,
            edge_edge_stiffness,
            edge_face_stiffness,
        )
        if not all(math.isfinite(value) for value in values) or radius_m <= 0.0:
            raise ValueError("Self-contact continuation values must be finite with positive radius")
        if min(values[1:]) < 0.0:
            raise ValueError("Self-contact continuation stiffness cannot be negative")
        self.solver.collision.radius = radius_m
        self.solver.collision.stiff_vf = vertex_face_stiffness
        self.solver.collision.stiff_ee = edge_edge_stiffness
        self.solver.collision.stiff_ef = edge_face_stiffness
        self.solver.rebuild_bvh(self.state_0)

    def center_of_mass_m(self) -> np.ndarray:
        positions = self.positions_m()
        mass = self._model_mass_kg
        return np.sum(positions * mass[:, None], axis=0) / np.sum(mass)

    def deformation_metrics(self) -> Mapping[str, float | int | bool]:
        positions = self.positions_m().astype(np.float64)
        velocity = self.velocities_m_s().astype(np.float64)
        world_triangles = positions[self.inputs.physical_triangles]
        material_triangles = self.inputs.grain_aligned_panel_coordinates_m[
            self.inputs.panel_triangles
        ]
        minimum_stretch = math.inf
        maximum_stretch = 0.0
        degenerate_count = 0
        for world, material in zip(world_triangles, material_triangles):
            dm = np.column_stack((material[1] - material[0], material[2] - material[0]))
            ds = np.column_stack((world[1] - world[0], world[2] - world[0]))
            singular = np.linalg.svd(ds @ np.linalg.inv(dm), compute_uv=False)
            minimum_stretch = min(minimum_stretch, float(np.min(singular)))
            maximum_stretch = max(maximum_stretch, float(np.max(singular)))
            if 0.5 * np.linalg.norm(np.cross(ds[:, 0], ds[:, 1])) <= 1.0e-12:
                degenerate_count += 1
        center = self.center_of_mass_m()
        drift = center - self.initial_center_of_mass_m
        mass = self._model_mass_kg
        total_linear_momentum = np.sum(velocity * mass[:, None], axis=0)
        center_velocity = total_linear_momentum / np.sum(mass)
        relative_position = positions - center
        relative_velocity = velocity - center_velocity
        total_angular_momentum = np.sum(
            np.cross(relative_position, relative_velocity) * mass[:, None],
            axis=0,
        )
        finite = bool(np.all(np.isfinite(positions)) and np.all(np.isfinite(velocity)))
        return {
            "step_count": self.step_count,
            "simulated_time_s": self.step_count * self.config.time_step_s,
            "finite": finite,
            "minimum_principal_stretch": minimum_stretch,
            "maximum_principal_stretch": maximum_stretch,
            "degenerate_triangle_count": degenerate_count,
            "maximum_speed_m_s": float(np.max(np.linalg.norm(velocity, axis=1))),
            "rms_speed_m_s": float(np.sqrt(np.mean(np.sum(velocity * velocity, axis=1)))),
            "center_of_mass_drift_m": float(np.linalg.norm(drift)),
            "center_of_mass_velocity_m_s": float(np.linalg.norm(center_velocity)),
            "linear_momentum_kg_m_s": float(np.linalg.norm(total_linear_momentum)),
            "angular_momentum_kg_m2_s": float(np.linalg.norm(total_angular_momentum)),
            "total_mass_kg": self.inputs.total_mass_kg,
        }

    def audit_summary(self) -> Mapping[str, Any]:
        model_mass = float(np.sum(self.model.particle_mass.numpy()))
        orientation = oriented_triangle_edge_audit(
            self.inputs.physical_triangles
        )
        return {
            "construction_id": self.construction.id,
            "newton_version": self.newton.__version__,
            "device": self.config.device,
            "particle_count": self.model.particle_count,
            "material_vertex_count": self.inputs.material_vertex_count,
            "triangle_count": self.model.tri_count,
            "edge_count": self.model.edge_count,
            "maximum_projective_neighbor_count": int(
                np.max(self.projective_neighbor_counts)
            ),
            "oriented_edge_audit": orientation,
            "authored_mass_kg": self.construction.total_panel_mass_kg,
            "adapter_mass_kg": self.inputs.total_mass_kg,
            "model_mass_kg": model_mass,
            "mass_error_kg": model_mass - self.construction.total_panel_mass_kg,
            "assembly_relaxation_applied": self.topology.relaxation.applied,
            "assembly_relaxation_optimizer_converged": self.topology.relaxation.optimizer_converged,
            "assembly_relaxation_iterations": self.topology.relaxation.iterations,
            "assembly_initial_edge_energy": self.topology.relaxation.initial_edge_energy,
            "assembly_final_edge_energy": self.topology.relaxation.final_edge_energy,
            "assembly_minimum_principal_stretch": self.topology.strain.minimum_principal_stretch,
            "assembly_maximum_principal_stretch": self.topology.strain.maximum_principal_stretch,
            "self_contact_enabled": self.config.enable_self_contact,
            "self_contact_vertex_face_stiffness": (
                self.solver.collision.stiff_vf if self.config.enable_self_contact else None
            ),
            "self_contact_edge_edge_stiffness": (
                self.solver.collision.stiff_ee if self.config.enable_self_contact else None
            ),
            "self_contact_edge_face_stiffness": (
                self.solver.collision.stiff_ef if self.config.enable_self_contact else None
            ),
            "self_contact_topological_filter_graph_hops": (
                self.config.self_contact_topological_filter_graph_hops
                if self.config.enable_self_contact
                else None
            ),
            "intersection_audit_local_exclusion_graph_hops": (
                self.config.intersection_audit_local_exclusion_graph_hops
            ),
            "intersection_audit_buffer_per_triangle": (
                self.config.intersection_audit_buffer_per_triangle
            ),
            "ipc_self_contact": self.ipc_self_contact_summary(),
            "table_plane_ccd": self.table_plane_ccd_summary(),
            "table_enabled": self.config.enable_table,
            "table_top_z_m": self.config.table_top_z_m if self.config.enable_table else None,
            "table_friction_mu": self.config.table_friction_mu if self.config.enable_table else None,
            "table_normal_projection_enabled": self.config.enable_table_normal_projection,
            "membrane_rate_damping_enabled": self.config.enable_membrane_rate_damping,
            "membrane_rate_damping_scale": (
                self.config.membrane_rate_damping_scale
                if self.config.enable_membrane_rate_damping
                else None
            ),
            "bending_rate_damping_enabled": self.config.enable_bending_rate_damping,
            "bending_rate_damping_scale": (
                self.config.bending_rate_damping_scale
                if self.config.enable_bending_rate_damping
                else None
            ),
            "quadratic_normal_air_drag_enabled": self.config.enable_air_drag,
            "air_density_kg_m3": (
                self.config.air_density_kg_m3 if self.config.enable_air_drag else None
            ),
            "normal_drag_coefficient": (
                self.config.normal_drag_coefficient if self.config.enable_air_drag else None
            ),
            "wind_velocity_m_s": (
                self.config.wind_velocity_m_s if self.config.enable_air_drag else None
            ),
            "structural_seam_hinge_count": int(
                np.sum(self.structural_seam_edge_mask)
            ),
            "structural_seam_bending_scale": (
                self.config.structural_seam_bending_scale
            ),
            "free_rigid_frame_gauge_enabled": self.config.enforce_free_rigid_frame,
            "self_contact_radius_m": (
                self.config.self_contact_radius_m if self.config.enable_self_contact else None
            ),
            "calibration_status": self.inputs.calibration_status,
            "unsupported_terms": self.inputs.unsupported_terms,
        }


__all__ = [
    "Style3DGarmentInputs",
    "Style3DGarmentSimulation",
    "Style3DSolverConfig",
    "build_style3d_inputs",
    "oriented_triangle_edge_audit",
    "style3d_projective_neighbor_counts",
]

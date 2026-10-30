"""Task-independent metrics and the first canonical towel half-fold predicate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

try:
    from .mesh import RectangularMesh
except ImportError:
    from mesh import RectangularMesh


@dataclass(frozen=True)
class HalfFoldThresholds:
    edge_alignment_rms_m: float = 0.025
    edge_alignment_max_m: float = 0.060
    fold_line_workspace_rms_m: float = 0.030
    fold_line_shape_rms_m: float = 0.005
    footprint_length_max_fraction: float = 0.62
    layer_order_fraction: float = 0.97
    settled_max_speed_m_s: float = 0.012
    maximum_strain: float = 0.12
    calibration_status: str = "task priors; validate on real-fold distributions"

    def __post_init__(self) -> None:
        if min(
            self.edge_alignment_rms_m,
            self.edge_alignment_max_m,
            self.fold_line_workspace_rms_m,
            self.fold_line_shape_rms_m,
            self.footprint_length_max_fraction,
            self.layer_order_fraction,
            self.settled_max_speed_m_s,
            self.maximum_strain,
        ) <= 0.0:
            raise ValueError("Task thresholds must be positive")
        if self.footprint_length_max_fraction > 1.0 or self.layer_order_fraction > 1.0:
            raise ValueError("Fraction thresholds cannot exceed one")


@dataclass(frozen=True)
class HalfFoldMetrics:
    edge_alignment_rms_m: float
    edge_alignment_max_m: float
    # RMS displacement from the initially registered workspace line. This
    # includes both rigid table slip and crease-shape error for compatibility
    # with early LaundryFold traces.
    fold_line_rms_m: float
    fold_line_shape_rms_m: float
    fold_line_translation_m: float
    footprint_length_fraction: float
    layer_order_fraction: float
    settled_max_speed_m_s: float
    maximum_strain: float
    invalid_intersections: int
    invalid_pose_writes: int
    released: bool
    settled: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TaskResult:
    success: bool
    metrics: HalfFoldMetrics
    failed_gates: tuple[str, ...]
    potential: float


class HalfFoldEvaluator:
    """Evaluate a left-to-right half fold using registered material vertices."""

    def __init__(
        self,
        mesh: RectangularMesh,
        thickness_m: float,
        thresholds: HalfFoldThresholds | None = None,
    ):
        if thickness_m <= 0.0:
            raise ValueError("thickness_m must be positive")
        if mesh.nx % 2:
            raise ValueError("Half-fold evaluation requires an even nx")
        self.mesh = mesh
        self.thickness_m = thickness_m
        self.thresholds = thresholds or HalfFoldThresholds()
        self.mid_ix = mesh.nx // 2
        self.fold_line = np.asarray(
            [iy * (mesh.nx + 1) + self.mid_ix for iy in range(mesh.ny + 1)],
            dtype=np.int32,
        )
        # Pairs share y and equal absolute material x, excluding the fold line.
        left: list[int] = []
        right: list[int] = []
        for iy in range(mesh.ny + 1):
            row = iy * (mesh.nx + 1)
            for ix in range(self.mid_ix):
                left.append(row + ix)
                right.append(row + (mesh.nx - ix))
        self.left_pairs = np.asarray(left, dtype=np.int32)
        self.right_pairs = np.asarray(right, dtype=np.int32)

    def evaluate(
        self,
        positions_m,
        velocities_m_s,
        *,
        maximum_strain: float,
        invalid_intersections: int,
        invalid_pose_writes: int,
        released: bool,
        settled: bool,
    ) -> TaskResult:
        positions = np.asarray(positions_m, dtype=np.float64)
        velocities = np.asarray(velocities_m_s, dtype=np.float64)
        expected = (self.mesh.vertex_count, 3)
        if positions.shape != expected or velocities.shape != expected:
            raise ValueError(f"positions and velocities must have shape {expected}")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("Task state must be finite")
        if min(invalid_intersections, invalid_pose_writes) < 0:
            raise ValueError("Invalid-event counts cannot be negative")

        left_edge = positions[self.mesh.boundary["left"]]
        right_edge = positions[self.mesh.boundary["right"]]
        edge_xy = np.linalg.norm(left_edge[:, :2] - right_edge[:, :2], axis=1)
        edge_rms = float(np.sqrt(np.mean(edge_xy**2)))
        edge_max = float(np.max(edge_xy))

        initial_fold_xy = self.mesh.positions_m[self.fold_line, :2]
        fold_displacement = positions[self.fold_line, :2] - initial_fold_xy
        fold_translation_xy = np.mean(fold_displacement, axis=0)
        fold_shape_residual = fold_displacement - fold_translation_xy
        fold_error = np.linalg.norm(fold_displacement, axis=1)
        fold_shape_error = np.linalg.norm(fold_shape_residual, axis=1)
        fold_rms = float(np.sqrt(np.mean(fold_error**2)))
        fold_shape_rms = float(np.sqrt(np.mean(fold_shape_error**2)))
        fold_translation = float(np.linalg.norm(fold_translation_xy))

        x_extent = float(np.max(positions[:, 0]) - np.min(positions[:, 0]))
        footprint_fraction = x_extent / self.mesh.length_m
        layer_tolerance = 0.5 * self.thickness_m
        layer_order = positions[self.left_pairs, 2] >= (
            positions[self.right_pairs, 2] - layer_tolerance
        )
        layer_fraction = float(np.mean(layer_order))
        max_speed = float(np.max(np.linalg.norm(velocities, axis=1)))

        metrics = HalfFoldMetrics(
            edge_alignment_rms_m=edge_rms,
            edge_alignment_max_m=edge_max,
            fold_line_rms_m=fold_rms,
            fold_line_shape_rms_m=fold_shape_rms,
            fold_line_translation_m=fold_translation,
            footprint_length_fraction=footprint_fraction,
            layer_order_fraction=layer_fraction,
            settled_max_speed_m_s=max_speed,
            maximum_strain=float(maximum_strain),
            invalid_intersections=int(invalid_intersections),
            invalid_pose_writes=int(invalid_pose_writes),
            released=bool(released),
            settled=bool(settled),
        )
        t = self.thresholds
        gates = {
            "edge_alignment_rms": edge_rms <= t.edge_alignment_rms_m,
            "edge_alignment_max": edge_max <= t.edge_alignment_max_m,
            "fold_line_workspace": fold_rms <= t.fold_line_workspace_rms_m,
            "fold_line_shape": fold_shape_rms <= t.fold_line_shape_rms_m,
            "footprint": footprint_fraction <= t.footprint_length_max_fraction,
            "layer_order": layer_fraction >= t.layer_order_fraction,
            "settled_speed": max_speed <= t.settled_max_speed_m_s,
            "strain": maximum_strain <= t.maximum_strain,
            "intersection": invalid_intersections == 0,
            "pose_writes": invalid_pose_writes == 0,
            "released": released,
            "settled": settled,
        }
        failed = tuple(name for name, passed in gates.items() if not passed)

        # Bounded diagnostic potential. Success remains the conjunction above.
        edge_score = math.exp(-edge_rms / max(t.edge_alignment_rms_m, 1.0e-12))
        workspace_score = math.exp(
            -fold_rms / max(t.fold_line_workspace_rms_m, 1.0e-12)
        )
        fold_shape_score = math.exp(
            -fold_shape_rms / max(t.fold_line_shape_rms_m, 1.0e-12)
        )
        fold_score = 0.5 * (workspace_score + fold_shape_score)
        footprint_score = float(np.clip((1.0 - footprint_fraction) / 0.5, 0.0, 1.0))
        potential = float(
            0.30 * edge_score
            + 0.20 * fold_score
            + 0.20 * footprint_score
            + 0.30 * layer_fraction
        )
        return TaskResult(not failed, metrics, failed, potential)

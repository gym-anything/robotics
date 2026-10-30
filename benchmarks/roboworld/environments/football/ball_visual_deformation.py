"""Physics-owned reduced visual deformation for the football shell.

The rigid-body centre, orientation, inertia, and spherical collider remain owned
by PhysX.  This module maps already-computed contact compression coordinates to
the separate ``/Ball/Render`` child only.  It never modifies collision geometry
or rigid state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

from .football_math import Quat, Vec3, norm, quat_inverse_rotate, unit


Matrix3 = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ContactCompressionMode:
    """One contact-owned indentation coordinate and outward normal."""

    name: str
    compression_m: float
    normal_world: Vec3
    surface_compression_m: float | None = None


@dataclass(frozen=True)
class AppliedCompressionMode:
    name: str
    requested_compression_m: float
    applied_compression_m: float
    requested_surface_compression_m: float
    applied_surface_compression_m: float
    normal_ball_local: Vec3
    axial_scale: float
    transverse_scale: float


@dataclass(frozen=True)
class BallVisualDeformationState:
    translation_ball_local_m: Vec3
    deformation_gradient_ball_local: Matrix3
    determinant: float
    maximum_applied_compression_m: float
    minimum_axial_scale: float
    maximum_transverse_scale: float
    modes: tuple[AppliedCompressionMode, ...]


def _determinant_3x3(matrix: Matrix3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def compute_ball_visual_deformation(
    *,
    ball_radius_m: float,
    volume_coupling: float,
    ball_orientation_world_wxyz: Quat,
    modes: Iterable[ContactCompressionMode],
    maximum_compression_fraction: float = 0.85,
    minimum_axial_scale: float = 0.45,
) -> BallVisualDeformationState:
    """Map contact coordinates to an order-independent local affine mode.

    A single contact with outward normal ``n`` uses

    ``F = transverse I + (axial - transverse) n n^T``

    and translates the render shell so the contacting pole agrees with the live
    coarse geometry.  When no separate surface coordinate is supplied this is
    ``compression*n/2``: the untouched pole stays fixed while the contacting
    pole moves inward by the full compression.  A separate surface coordinate
    lets a subcycled constitutive mode remain visible while its contact pole is
    anchored to the current outer-step geometry.  Multiple contacts contribute
    symmetric strain increments additively.  This is a visual reduced mode, not
    a claim that the shell has a spatially resolved finite-element state.
    """

    if not math.isfinite(ball_radius_m) or ball_radius_m <= 0.0:
        raise ValueError("ball radius must be positive and finite")
    if not math.isfinite(volume_coupling) or not 0.0 <= volume_coupling <= 1.0:
        raise ValueError("volume coupling must be finite and in [0, 1]")
    if not 0.0 < maximum_compression_fraction < 1.0:
        raise ValueError("maximum compression fraction must be in (0, 1)")
    if not 0.0 < minimum_axial_scale <= 1.0:
        raise ValueError("minimum axial scale must be in (0, 1]")

    gradient = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    translation = [0.0, 0.0, 0.0]
    applied: list[AppliedCompressionMode] = []
    compression_limit = maximum_compression_fraction * ball_radius_m

    # Stable names make simultaneous-contact composition independent of caller
    # container order and give every contribution explicit provenance.
    for mode in sorted(modes, key=lambda item: item.name):
        if not mode.name:
            raise ValueError("compression mode name must be non-empty")
        if not math.isfinite(mode.compression_m) or mode.compression_m < 0.0:
            raise ValueError("contact compression must be non-negative and finite")
        surface_compression = (
            mode.compression_m
            if mode.surface_compression_m is None
            else mode.surface_compression_m
        )
        if (
            not math.isfinite(surface_compression)
            or surface_compression < 0.0
        ):
            raise ValueError(
                "surface compression must be non-negative and finite"
            )
        if not all(math.isfinite(value) for value in mode.normal_world):
            raise ValueError("contact normal must be finite")
        if mode.compression_m == 0.0:
            continue
        if norm(mode.normal_world) <= 1.0e-12:
            raise ValueError("active contact normal must be non-zero")

        compression = min(mode.compression_m, compression_limit)
        applied_surface_compression = min(
            surface_compression, compression_limit
        )
        normal_local = unit(
            quat_inverse_rotate(
                ball_orientation_world_wxyz,
                unit(mode.normal_world),
            )
        )
        axial = max(
            minimum_axial_scale,
            1.0 - compression / (2.0 * ball_radius_m),
        )
        # Only the pressure model's effective volume fraction produces the
        # transverse bulge.  Full incompressibility visibly overstates it.
        transverse = 1.0 + 0.5 * volume_coupling * (
            1.0 / math.sqrt(axial) - 1.0
        )

        for row in range(3):
            translation[row] += (
                applied_surface_compression - 0.5 * compression
            ) * normal_local[row]
            for column in range(3):
                identity = 1.0 if row == column else 0.0
                mode_value = (
                    transverse * identity
                    + (axial - transverse)
                    * normal_local[row]
                    * normal_local[column]
                )
                gradient[row][column] += mode_value - identity

        applied.append(
            AppliedCompressionMode(
                name=mode.name,
                requested_compression_m=mode.compression_m,
                applied_compression_m=compression,
                requested_surface_compression_m=surface_compression,
                applied_surface_compression_m=applied_surface_compression,
                normal_ball_local=normal_local,
                axial_scale=axial,
                transverse_scale=transverse,
            )
        )

    frozen_gradient: Matrix3 = tuple(
        tuple(float(value) for value in row) for row in gradient
    )
    determinant = _determinant_3x3(frozen_gradient)
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise ValueError(
            "combined visual deformation is inverted; reduce contact modes"
        )

    return BallVisualDeformationState(
        translation_ball_local_m=tuple(translation),  # type: ignore[arg-type]
        deformation_gradient_ball_local=frozen_gradient,
        determinant=determinant,
        maximum_applied_compression_m=max(
            (mode.applied_compression_m for mode in applied), default=0.0
        ),
        minimum_axial_scale=min(
            (mode.axial_scale for mode in applied), default=1.0
        ),
        maximum_transverse_scale=max(
            (mode.transverse_scale for mode in applied), default=1.0
        ),
        modes=tuple(applied),
    )


class UsdBallVisualDeformationAdapter:
    """Apply the pure deformation state to a visual-only USD child."""

    def __init__(
        self,
        stage,
        *,
        render_path: str,
        collider_path: str,
        ball_radius_m: float,
        volume_coupling: float,
    ) -> None:
        # Keep pxr out of the module import path so dependency-light unit tests
        # can exercise the mapping without constructing SimulationApp.
        from pxr import Gf, Sdf, UsdGeom

        render_prim = stage.GetPrimAtPath(render_path)
        collider_prim = stage.GetPrimAtPath(collider_path)
        if not render_prim.IsValid() or not collider_prim.IsValid():
            raise ValueError("ball render and collider prims must exist")

        self._Gf = Gf
        self._UsdGeom = UsdGeom
        self.render_path = render_path
        self.collider_path = collider_path
        self.ball_radius_m = ball_radius_m
        self.volume_coupling = volume_coupling
        self._render_xform = UsdGeom.Xformable(render_prim)
        self._render_xform.ClearXformOpOrder()
        self._transform_op = self._render_xform.AddTransformOp()
        self._collider_xform = UsdGeom.Xformable(collider_prim)
        self._collider_ops_initial = tuple(
            operation.GetOpName()
            for operation in self._collider_xform.GetOrderedXformOps()
        )
        self._collider_radius_initial = float(
            UsdGeom.Sphere(collider_prim).GetRadiusAttr().Get()
        )

        render_prim.CreateAttribute(
            "football:visualDeformationOwner",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set("contact compression coordinates; render child only")
        render_prim.CreateAttribute(
            "football:collisionGeometryAffected",
            Sdf.ValueTypeNames.Bool,
            custom=True,
        ).Set(False)
        render_prim.CreateAttribute(
            "football:deformationModelStatus",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set("REDUCED VISUAL MODE / PRIOR-ONLY / NOT SHELL FEA")

        self.update_count = 0
        self.deformed_update_count = 0
        self.peak_compression_by_source_m: dict[str, float] = {}
        self.peak_surface_compression_by_source_m: dict[str, float] = {}
        self.maximum_modal_surface_disagreement_m = 0.0
        self.minimum_determinant = 1.0
        self.minimum_axial_scale_seen = 1.0
        self.maximum_transverse_scale_seen = 1.0
        self.maximum_translation_m = 0.0
        self.last_state = compute_ball_visual_deformation(
            ball_radius_m=ball_radius_m,
            volume_coupling=volume_coupling,
            ball_orientation_world_wxyz=(1.0, 0.0, 0.0, 0.0),
            modes=(),
        )
        self._apply_state(self.last_state)

    def _apply_state(self, state: BallVisualDeformationState) -> None:
        gradient = state.deformation_gradient_ball_local
        translation = state.translation_ball_local_m
        matrix = self._Gf.Matrix4d(
            gradient[0][0], gradient[0][1], gradient[0][2], 0.0,
            gradient[1][0], gradient[1][1], gradient[1][2], 0.0,
            gradient[2][0], gradient[2][1], gradient[2][2], 0.0,
            translation[0], translation[1], translation[2], 1.0,
        )
        self._transform_op.Set(matrix)

    def apply(
        self,
        *,
        ball_orientation_world_wxyz: Quat,
        modes: Iterable[ContactCompressionMode],
    ) -> BallVisualDeformationState:
        state = compute_ball_visual_deformation(
            ball_radius_m=self.ball_radius_m,
            volume_coupling=self.volume_coupling,
            ball_orientation_world_wxyz=ball_orientation_world_wxyz,
            modes=modes,
        )
        self._apply_state(state)
        self.last_state = state
        self.update_count += 1
        if state.modes:
            self.deformed_update_count += 1
        for mode in state.modes:
            self.peak_compression_by_source_m[mode.name] = max(
                self.peak_compression_by_source_m.get(mode.name, 0.0),
                mode.applied_compression_m,
            )
            self.peak_surface_compression_by_source_m[mode.name] = max(
                self.peak_surface_compression_by_source_m.get(mode.name, 0.0),
                mode.applied_surface_compression_m,
            )
            self.maximum_modal_surface_disagreement_m = max(
                self.maximum_modal_surface_disagreement_m,
                abs(
                    mode.applied_compression_m
                    - mode.applied_surface_compression_m
                ),
            )
        self.minimum_determinant = min(
            self.minimum_determinant, state.determinant
        )
        self.minimum_axial_scale_seen = min(
            self.minimum_axial_scale_seen, state.minimum_axial_scale
        )
        self.maximum_transverse_scale_seen = max(
            self.maximum_transverse_scale_seen, state.maximum_transverse_scale
        )
        self.maximum_translation_m = max(
            self.maximum_translation_m,
            norm(state.translation_ball_local_m),
        )
        return state

    def report(self) -> dict[str, object]:
        collider_ops_final = tuple(
            operation.GetOpName()
            for operation in self._collider_xform.GetOrderedXformOps()
        )
        collider_radius_final = float(
            self._UsdGeom.Sphere(
                self._collider_xform.GetPrim()
            ).GetRadiusAttr().Get()
        )
        collider_unchanged = (
            collider_ops_final == self._collider_ops_initial
            and collider_radius_final == self._collider_radius_initial
        )
        return {
            "model_status": "REDUCED VISUAL MODE / PRIOR-ONLY / NOT SHELL FEA",
            "owner": "contact compression coordinates",
            "render_path": self.render_path,
            "collider_path": self.collider_path,
            "update_count": self.update_count,
            "deformed_update_count": self.deformed_update_count,
            "peak_compression_by_source_m": dict(
                sorted(self.peak_compression_by_source_m.items())
            ),
            "peak_surface_compression_by_source_m": dict(
                sorted(self.peak_surface_compression_by_source_m.items())
            ),
            "maximum_modal_surface_disagreement_m": (
                self.maximum_modal_surface_disagreement_m
            ),
            "minimum_deformation_determinant": self.minimum_determinant,
            "minimum_axial_scale": self.minimum_axial_scale_seen,
            "maximum_transverse_scale": self.maximum_transverse_scale_seen,
            "maximum_translation_m": self.maximum_translation_m,
            "last_state": asdict(self.last_state),
            "collider_xform_ops_initial": self._collider_ops_initial,
            "collider_xform_ops_final": collider_ops_final,
            "collider_radius_initial_m": self._collider_radius_initial,
            "collider_radius_final_m": collider_radius_final,
            "collider_unchanged": collider_unchanged,
        }

"""Authoritative dimensions, evidence labels, and numerical profiles.

All values are SI.  The canonical towel numbers are deliberately labelled
``engineering-prior``: they make the software executable before a named towel
is acquired, but they are not measurements and cannot support a real-world
fidelity claim.  ``TextileProfile`` keeps the physical quantities separate from
solver knobs so numerical tuning cannot silently become material calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = "laundry-specs/1.0"
ENVIRONMENT_VERSION = "0.1.0-dev"


@dataclass(frozen=True)
class ParameterEvidence:
    value: float
    unit: str
    status: str
    source: str
    note: str = ""


PARAMETERS: Mapping[str, ParameterEvidence] = {
    "gravity": ParameterEvidence(
        9.80665,
        "m/s^2",
        "conventional-standard",
        "CGPM conventional standard gravity",
    ),
    "towel_length": ParameterEvidence(
        0.60,
        "m",
        "engineering-prior",
        "canonical software coupon",
        "Replace with conditioned specimen measurement.",
    ),
    "towel_width": ParameterEvidence(
        0.40,
        "m",
        "engineering-prior",
        "canonical software coupon",
        "Replace with conditioned specimen measurement.",
    ),
    "towel_areal_density": ParameterEvidence(
        0.30,
        "kg/m^2",
        "engineering-prior",
        "plausible woven-cotton initialization",
        "Measure according to the recorded ASTM D3776/D3776M procedure.",
    ),
    "towel_unloaded_thickness": ParameterEvidence(
        0.80e-3,
        "m",
        "engineering-prior",
        "plausible plain-woven towel initialization",
        "Measure thickness over pressure; one unloaded number is insufficient.",
    ),
    "warp_membrane_stiffness": ParameterEvidence(
        8.0e3,
        "N/m",
        "numerical-prior",
        "Newton Style3D example scale and cloth-literature model form",
        "Fit the full nonlinear curve from warp cyclic coupons.",
    ),
    "weft_membrane_stiffness": ParameterEvidence(
        6.0e3,
        "N/m",
        "numerical-prior",
        "Newton Style3D example scale and cloth-literature model form",
        "Fit the full nonlinear curve from weft cyclic coupons.",
    ),
    "bias_shear_stiffness": ParameterEvidence(
        4.0e2,
        "N/m",
        "numerical-prior",
        "anisotropic shell initialization",
        "Fit from bias/shear coupons; do not infer from tensile modulus.",
    ),
    "warp_bending_stiffness": ParameterEvidence(
        2.0e-5,
        "N m",
        "numerical-prior",
        "Newton Style3D cloth example scale",
        "Fit moment-curvature response with mesh-independent conversion.",
    ),
    "weft_bending_stiffness": ParameterEvidence(
        1.5e-5,
        "N m",
        "numerical-prior",
        "Newton Style3D cloth example scale",
        "Fit moment-curvature response with mesh-independent conversion.",
    ),
    "bias_bending_stiffness": ParameterEvidence(
        1.0e-5,
        "N m",
        "numerical-prior",
        "Newton Style3D cloth example scale",
        "Fit moment-curvature response with mesh-independent conversion.",
    ),
}


def _p(name: str) -> float:
    return PARAMETERS[name].value


@dataclass(frozen=True)
class TowelSpec:
    """Canonical rectangular coupon/garment before a real specimen is selected."""

    length_m: float = _p("towel_length")
    width_m: float = _p("towel_width")
    areal_density_kg_m2: float = _p("towel_areal_density")
    unloaded_thickness_m: float = _p("towel_unloaded_thickness")
    specimen_id: str = "PRIOR-COTTON-TOWEL-000"
    calibration_status: str = "prior-only / not physically calibrated"

    def __post_init__(self) -> None:
        if min(
            self.length_m,
            self.width_m,
            self.areal_density_kg_m2,
            self.unloaded_thickness_m,
        ) <= 0.0:
            raise ValueError("Towel dimensions, density, and thickness must be positive")
        if self.length_m <= self.width_m:
            raise ValueError("Canonical towel length must exceed its width")
        if not self.specimen_id:
            raise ValueError("Every towel needs a specimen ID")

    @property
    def area_m2(self) -> float:
        return self.length_m * self.width_m

    @property
    def mass_kg(self) -> float:
        return self.area_m2 * self.areal_density_kg_m2


@dataclass(frozen=True)
class TextileProfile:
    """Material-space model parameters, not engine iteration settings."""

    name: str
    warp_membrane_n_m: float
    weft_membrane_n_m: float
    shear_n_m: float
    warp_tension_stiffening_n_m: float
    weft_tension_stiffening_n_m: float
    cross_coupling_ratio: float
    warp_bend_nm: float
    weft_bend_nm: float
    bias_bend_nm: float
    membrane_damping_n_s_m: float
    bend_damping_nm_s: float
    table_static_friction: float
    table_kinetic_friction: float
    cloth_static_friction: float
    cloth_kinetic_friction: float
    finger_static_friction: float
    finger_kinetic_friction: float
    minimum_compressed_thickness_m: float
    compression_pressure_scale_pa: float
    calibration_status: str

    def __post_init__(self) -> None:
        positive = (
            self.warp_membrane_n_m,
            self.weft_membrane_n_m,
            self.shear_n_m,
            self.warp_bend_nm,
            self.weft_bend_nm,
            self.bias_bend_nm,
            self.minimum_compressed_thickness_m,
            self.compression_pressure_scale_pa,
        )
        if min(positive) <= 0.0:
            raise ValueError("Textile stiffness, thickness, and pressure scale must be positive")
        if self.membrane_damping_n_s_m < 0.0 or self.bend_damping_nm_s < 0.0:
            raise ValueError("Textile damping coefficients cannot be negative")
        if not 0.0 <= self.cross_coupling_ratio < 1.0:
            raise ValueError("Cross-coupling ratio must lie in [0, 1)")
        for value in (
            self.table_static_friction,
            self.table_kinetic_friction,
            self.cloth_static_friction,
            self.cloth_kinetic_friction,
            self.finger_static_friction,
            self.finger_kinetic_friction,
        ):
            if value < 0.0:
                raise ValueError("Friction coefficients cannot be negative")
        if self.table_kinetic_friction > self.table_static_friction:
            raise ValueError("Table kinetic friction cannot exceed static friction")
        if self.cloth_kinetic_friction > self.cloth_static_friction:
            raise ValueError("Cloth kinetic friction cannot exceed static friction")
        if self.finger_kinetic_friction > self.finger_static_friction:
            raise ValueError("Finger kinetic friction cannot exceed static friction")


MATERIAL_PROFILES: Mapping[str, TextileProfile] = {
    "cotton_towel_prior": TextileProfile(
        name="cotton_towel_prior",
        warp_membrane_n_m=_p("warp_membrane_stiffness"),
        weft_membrane_n_m=_p("weft_membrane_stiffness"),
        shear_n_m=_p("bias_shear_stiffness"),
        warp_tension_stiffening_n_m=2.0e5,
        weft_tension_stiffening_n_m=1.5e5,
        cross_coupling_ratio=0.08,
        warp_bend_nm=_p("warp_bending_stiffness"),
        weft_bend_nm=_p("weft_bending_stiffness"),
        bias_bend_nm=_p("bias_bending_stiffness"),
        membrane_damping_n_s_m=0.05,
        bend_damping_nm_s=1.0e-6,
        table_static_friction=0.50,
        table_kinetic_friction=0.40,
        cloth_static_friction=0.55,
        cloth_kinetic_friction=0.44,
        finger_static_friction=0.90,
        finger_kinetic_friction=0.75,
        minimum_compressed_thickness_m=0.35e-3,
        compression_pressure_scale_pa=4.0e3,
        calibration_status="prior-only / not physically calibrated",
    ),
    "cotton_jersey_prior": TextileProfile(
        name="cotton_jersey_prior",
        warp_membrane_n_m=5.0e2,
        weft_membrane_n_m=7.0e2,
        shear_n_m=8.0e1,
        warp_tension_stiffening_n_m=2.0e4,
        weft_tension_stiffening_n_m=2.8e4,
        cross_coupling_ratio=0.12,
        warp_bend_nm=2.0e-6,
        weft_bend_nm=2.5e-6,
        bias_bend_nm=1.5e-6,
        membrane_damping_n_s_m=0.035,
        bend_damping_nm_s=2.0e-7,
        table_static_friction=0.46,
        table_kinetic_friction=0.36,
        cloth_static_friction=0.52,
        cloth_kinetic_friction=0.41,
        finger_static_friction=0.86,
        finger_kinetic_friction=0.70,
        minimum_compressed_thickness_m=0.25e-3,
        compression_pressure_scale_pa=2.5e3,
        calibration_status="material-family prior only; no jersey coupon measured",
    ),
    "cotton_poplin_prior": TextileProfile(
        name="cotton_poplin_prior",
        warp_membrane_n_m=6.0e3,
        weft_membrane_n_m=4.0e3,
        shear_n_m=3.0e2,
        warp_tension_stiffening_n_m=1.3e5,
        weft_tension_stiffening_n_m=9.0e4,
        cross_coupling_ratio=0.07,
        warp_bend_nm=8.0e-6,
        weft_bend_nm=5.0e-6,
        bias_bend_nm=4.0e-6,
        membrane_damping_n_s_m=0.040,
        bend_damping_nm_s=5.0e-7,
        table_static_friction=0.42,
        table_kinetic_friction=0.33,
        cloth_static_friction=0.48,
        cloth_kinetic_friction=0.38,
        finger_static_friction=0.82,
        finger_kinetic_friction=0.67,
        minimum_compressed_thickness_m=0.15e-3,
        compression_pressure_scale_pa=5.0e3,
        calibration_status="material-family prior only; no poplin coupon measured",
    ),
    "denim_twill_prior": TextileProfile(
        name="denim_twill_prior",
        warp_membrane_n_m=1.4e4,
        weft_membrane_n_m=1.0e4,
        shear_n_m=1.0e3,
        warp_tension_stiffening_n_m=3.0e5,
        weft_tension_stiffening_n_m=2.2e5,
        cross_coupling_ratio=0.10,
        warp_bend_nm=1.2e-4,
        weft_bend_nm=9.0e-5,
        bias_bend_nm=6.0e-5,
        membrane_damping_n_s_m=0.080,
        bend_damping_nm_s=4.0e-6,
        table_static_friction=0.55,
        table_kinetic_friction=0.44,
        cloth_static_friction=0.62,
        cloth_kinetic_friction=0.50,
        finger_static_friction=0.94,
        finger_kinetic_friction=0.78,
        minimum_compressed_thickness_m=0.45e-3,
        compression_pressure_scale_pa=1.2e4,
        calibration_status="material-family prior only; no denim coupon measured",
    ),
    "terry_cotton_prior": TextileProfile(
        name="terry_cotton_prior",
        warp_membrane_n_m=4.5e3,
        weft_membrane_n_m=3.5e3,
        shear_n_m=3.0e2,
        warp_tension_stiffening_n_m=1.0e5,
        weft_tension_stiffening_n_m=8.0e4,
        cross_coupling_ratio=0.09,
        warp_bend_nm=3.5e-5,
        weft_bend_nm=3.0e-5,
        bias_bend_nm=2.2e-5,
        membrane_damping_n_s_m=0.075,
        bend_damping_nm_s=2.5e-6,
        table_static_friction=0.62,
        table_kinetic_friction=0.49,
        cloth_static_friction=0.72,
        cloth_kinetic_friction=0.58,
        finger_static_friction=1.00,
        finger_kinetic_friction=0.83,
        minimum_compressed_thickness_m=1.0e-3,
        compression_pressure_scale_pa=2.5e3,
        calibration_status="material-family prior only; no terry pile coupon measured",
    ),
    "viscose_crepe_prior": TextileProfile(
        name="viscose_crepe_prior",
        warp_membrane_n_m=1.8e3,
        weft_membrane_n_m=1.5e3,
        shear_n_m=1.2e2,
        warp_tension_stiffening_n_m=5.0e4,
        weft_tension_stiffening_n_m=4.0e4,
        cross_coupling_ratio=0.06,
        warp_bend_nm=3.0e-6,
        weft_bend_nm=2.5e-6,
        bias_bend_nm=1.8e-6,
        membrane_damping_n_s_m=0.045,
        bend_damping_nm_s=3.0e-7,
        table_static_friction=0.48,
        table_kinetic_friction=0.37,
        cloth_static_friction=0.56,
        cloth_kinetic_friction=0.44,
        finger_static_friction=0.84,
        finger_kinetic_friction=0.68,
        minimum_compressed_thickness_m=0.12e-3,
        compression_pressure_scale_pa=3.5e3,
        calibration_status="material-family prior only; no crepe coupon measured",
    ),
    "cotton_rib_prior": TextileProfile(
        name="cotton_rib_prior",
        warp_membrane_n_m=8.0e2,
        weft_membrane_n_m=1.2e3,
        shear_n_m=9.0e1,
        warp_tension_stiffening_n_m=3.0e4,
        weft_tension_stiffening_n_m=5.0e4,
        cross_coupling_ratio=0.15,
        warp_bend_nm=3.0e-6,
        weft_bend_nm=4.0e-6,
        bias_bend_nm=2.0e-6,
        membrane_damping_n_s_m=0.050,
        bend_damping_nm_s=4.0e-7,
        table_static_friction=0.50,
        table_kinetic_friction=0.39,
        cloth_static_friction=0.60,
        cloth_kinetic_friction=0.47,
        finger_static_friction=0.90,
        finger_kinetic_friction=0.74,
        minimum_compressed_thickness_m=0.30e-3,
        compression_pressure_scale_pa=3.0e3,
        calibration_status="material-family prior only; no rib-knit coupon measured",
    ),
}


@dataclass(frozen=True)
class PhysicsProfile:
    name: str
    cloth_nx: int
    cloth_ny: int
    frame_hz: int
    substeps: int
    solver_iterations: int
    contact_margin_m: float
    self_contact_detection_margin_m: float
    self_contact_conservative_relaxation: float = 0.85
    enable_full_surface_contact: bool = True
    calibration_status: str = "numerical profile; convergence not yet established"

    def __post_init__(self) -> None:
        if min(
            self.cloth_nx,
            self.cloth_ny,
            self.frame_hz,
            self.substeps,
            self.solver_iterations,
        ) <= 0:
            raise ValueError("Physics profile counts/rates must be positive")
        if self.contact_margin_m < 0.0 or self.self_contact_detection_margin_m <= 0.0:
            raise ValueError("Contact margins must be nonnegative/positive")
        if not 0.0 < self.self_contact_conservative_relaxation <= 1.0:
            raise ValueError("Self-contact conservative relaxation must lie in (0, 1]")

    @property
    def frame_dt_s(self) -> float:
        return 1.0 / self.frame_hz

    @property
    def physics_dt_s(self) -> float:
        return self.frame_dt_s / self.substeps

    @property
    def maximum_self_contact_sweep_speed_m_s(self) -> float:
        """Largest displacement rate not clipped by self-contact CCD.

        Newton's penetration-free self-contact path limits one-substep motion to
        ``relaxation * detection_margin``.  Publishing this ceiling prevents a
        small query margin from silently becoming an artificial terminal speed.
        """

        return (
            self.self_contact_conservative_relaxation
            * self.self_contact_detection_margin_m
            / self.physics_dt_s
        )


PHYSICS_PROFILES: Mapping[str, PhysicsProfile] = {
    # The self-contact detection margin is a swept-collision query distance,
    # not the physical cloth thickness.  These values keep the published CCD
    # speed ceiling above a 100 mm gravity-drop impact while shrinking under
    # timestep refinement.
    # At 240 Hz a second penalty-contact impact becomes unstable.  The
    # retained coarse mesh therefore runs at 480 Hz.
    "fast": PhysicsProfile("fast", 24, 16, 60, 8, 6, 1.5e-3, 5.0e-3),
    # 480 Hz fails for the 48 x 32 mesh under a tight registered fold;
    # 960 Hz is the first retained rate in the exercised sweep.
    "interactive": PhysicsProfile("interactive", 48, 32, 60, 16, 10, 1.0e-3, 3.0e-3),
    # The validation step scales approximately with cell width from the
    # retained interactive case (1,440 Hz for the 72 x 48 mesh).
    "validation": PhysicsProfile("validation", 72, 48, 60, 24, 20, 0.6e-3, 2.0e-3),
}


def json_ready(value):
    if hasattr(value, "__dataclass_fields__"):
        return {key: json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_spec_manifest(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "parameters": json_ready(PARAMETERS),
        "towel": json_ready(TowelSpec()),
        "materials": json_ready(MATERIAL_PROFILES),
        "physics_profiles": json_ready(PHYSICS_PROFILES),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output

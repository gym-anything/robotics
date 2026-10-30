"""Authoritative football dimensions, evidence labels, and numerical profiles.

Everything is SI.  A value tagged ``law`` or ``standard`` is fixed by a cited
definition; ``measured-range`` is bounded by experiments on real balls; and
``prior`` is only an initial model parameter awaiting calibration.  The labels
are exported with traces and USD metadata so a good-looking trajectory cannot
silently turn an engineering prior into a measured fact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ParameterEvidence:
    value: float
    unit: str
    status: str
    source: str
    note: str = ""


PARAMETERS: Mapping[str, ParameterEvidence] = {
    "ball_circumference": ParameterEvidence(
        0.690, "m", "standard-range midpoint", "IFAB Laws of the Game 2025/26, Law 2",
        "Law 2 range is 0.68--0.70 m; replace with the measured test ball.",
    ),
    "ball_mass": ParameterEvidence(
        0.430, "kg", "standard-range midpoint", "IFAB Laws of the Game 2025/26, Law 2",
        "Law 2 range is 0.410--0.450 kg at the start of the match.",
    ),
    "goal_inside_width": ParameterEvidence(
        7.32, "m", "standard", "IFAB Laws of the Game 2025/26, Law 1",
    ),
    "goal_inside_height": ParameterEvidence(
        2.44, "m", "standard", "IFAB Laws of the Game 2025/26, Law 1",
    ),
    "goal_post_diameter": ParameterEvidence(
        0.120, "m", "standard maximum", "IFAB Laws of the Game 2025/26, Law 1",
        "The initial frame uses the maximum legal circular section.",
    ),
    "penalty_mark_distance": ParameterEvidence(
        11.0, "m", "standard", "IFAB Laws of the Game 2025/26, Law 1",
    ),
    "gravity": ParameterEvidence(
        9.80665, "m/s^2", "conventional standard", "CGPM conventional standard gravity",
    ),
    "drag_subcritical": ParameterEvidence(
        0.50, "1", "prior", "Asai et al. and Passmore et al. curve-shape initialization",
        "Generic 32-panel ball only; fit a measured Cd(Re,S,orientation) surface.",
    ),
    "drag_supercritical": ParameterEvidence(
        0.20, "1", "prior", "Asai et al. and Passmore et al. curve-shape initialization",
        "Generic 32-panel ball only; not a branded-ball calibration.",
    ),
    "drag_critical_reynolds": ParameterEvidence(
        250_000.0, "1", "prior", "wind-tunnel literature scale",
        "Panel geometry and surface roughness move the drag crisis substantially.",
    ),
    "drag_transition_width": ParameterEvidence(
        25_000.0, "1", "prior", "smooth numerical initialization",
    ),
    "magnus_slope": ParameterEvidence(
        1.20, "1", "prior", "soccer-ball flight literature scale",
        "Slope maps spin parameter S to lift coefficient before saturation.",
    ),
    "magnus_cap": ParameterEvidence(
        0.32, "1", "prior", "soccer-ball flight literature scale",
    ),
    "spin_decay_per_m": ParameterEvidence(
        0.005, "1/m", "prior", "James and Haake spin-decay model scale",
        "Used in d(omega)/dt=-k*|u|*omega; requires spin-tracking calibration.",
    ),
    "knuckle_coefficient_cap": ParameterEvidence(
        0.14, "1", "prior", "Hong and Asai unsteady-force scale",
        "Stateful wake amplitude, never independent random force and never target-selected.",
    ),
    "knuckle_reference_frequency": ParameterEvidence(
        3.5, "Hz", "prior", "unsteady football-force literature scale near 25 m/s",
    ),
}


def _value(name: str) -> float:
    return PARAMETERS[name].value


@dataclass(frozen=True)
class BallSpec:
    """Nominal regulation-size generic 32-panel football."""

    mass_kg: float = _value("ball_mass")
    circumference_m: float = _value("ball_circumference")
    groove_depth_m: float = 1.5e-3
    shell_thickness_m: float = 3.0e-3
    restitution_prior: float = 0.72
    static_friction_prior: float = 0.55
    dynamic_friction_prior: float = 0.42

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0 or self.circumference_m <= 0.0:
            raise ValueError("ball mass and circumference must be positive")
        if not 0.0 <= self.restitution_prior <= 1.0:
            raise ValueError("restitution must be in [0, 1]")
        if not 0.0 <= self.groove_depth_m < self.radius_m:
            raise ValueError("groove depth must lie inside the ball radius")

    @property
    def diameter_m(self) -> float:
        return self.circumference_m / math.pi

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m

    @property
    def projected_area_m2(self) -> float:
        return math.pi * self.radius_m**2

    @property
    def displaced_volume_m3(self) -> float:
        return 4.0 * math.pi * self.radius_m**3 / 3.0

    @property
    def thin_shell_inertia_kg_m2(self) -> float:
        # The production value must be measured; 2/3 m r^2 is the transparent
        # thin-shell prior and materially differs from a solid-sphere default.
        return 2.0 * self.mass_kg * self.radius_m**2 / 3.0


@dataclass(frozen=True)
class AtmosphereSpec:
    temperature_k: float = 293.15
    pressure_pa: float = 101_325.0
    relative_humidity: float = 0.50
    wind_velocity_world_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.temperature_k <= 0.0 or self.pressure_pa <= 0.0:
            raise ValueError("temperature and pressure must be positive")
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError("relative humidity must be in [0, 1]")


@dataclass(frozen=True)
class AerodynamicPrior:
    drag_subcritical: float = _value("drag_subcritical")
    drag_supercritical: float = _value("drag_supercritical")
    drag_critical_reynolds: float = _value("drag_critical_reynolds")
    drag_transition_width: float = _value("drag_transition_width")
    drag_spin_gain: float = 0.08
    drag_seam_modulation: float = 0.035
    magnus_slope: float = _value("magnus_slope")
    magnus_cap: float = _value("magnus_cap")
    reverse_magnus_amplitude: float = 0.20
    reverse_magnus_reynolds: float = 210_000.0
    reverse_magnus_reynolds_width: float = 38_000.0
    reverse_magnus_spin: float = 0.135
    reverse_magnus_spin_width: float = 0.055
    spin_decay_per_m: float = _value("spin_decay_per_m")
    knuckle_coefficient_cap: float = _value("knuckle_coefficient_cap")
    knuckle_reference_frequency_hz: float = _value("knuckle_reference_frequency")
    knuckle_reference_speed_m_s: float = 25.0
    knuckle_spin_scale: float = 0.055
    wake_relaxation_time_s: float = 0.080
    wake_vertical_fraction: float = 0.55
    minimum_aero_speed_m_s: float = 0.05


@dataclass(frozen=True)
class FieldSpec:
    goal_plane_x_m: float = 0.0
    penalty_spot_x_m: float = -_value("penalty_mark_distance")
    goal_inside_width_m: float = _value("goal_inside_width")
    goal_inside_height_m: float = _value("goal_inside_height")
    goal_post_diameter_m: float = _value("goal_post_diameter")
    pitch_length_m: float = 34.0
    pitch_width_m: float = 24.0
    goal_depth_m: float = 2.1
    line_width_m: float = 0.10
    grass_height_m: float = 0.028

    @property
    def half_goal_width_m(self) -> float:
        return 0.5 * self.goal_inside_width_m


@dataclass(frozen=True)
class PhysicsProfile:
    name: str
    physics_hz: int
    render_hz: int
    solver_position_iterations: int
    solver_velocity_iterations: int
    use_gpu_dynamics: bool = True
    use_gpu_broadphase: bool = True
    enable_scene_ccd: bool = True
    enable_body_ccd: bool = True
    enable_speculative_ccd: bool = True

    @property
    def physics_dt_s(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def rendering_dt_s(self) -> float:
        return 1.0 / self.render_hz


PHYSICS_PROFILES: Mapping[str, PhysicsProfile] = {
    "fast": PhysicsProfile("fast", 240, 60, 24, 4),
    "interactive": PhysicsProfile("interactive", 480, 60, 32, 8),
    "validation": PhysicsProfile("validation", 960, 60, 48, 12, False, False),
    "impact_reference": PhysicsProfile("impact_reference", 2000, 50, 64, 16, False, False),
    # A regulation kick's force pulse is only about 9 ms.  Twenty kilohertz
    # gives roughly 170 samples through contact and keeps the eccentric
    # work/energy defect below one percent of pre-impact kinetic energy.
    # The 2 kHz profile remains a responsive visualisation convenience only.
    "impact_validation": PhysicsProfile("impact_validation", 20000, 50, 96, 24, False, False),
    "surface_validation": PhysicsProfile("surface_validation", 20000, 50, 64, 16, False, False),
    "net_validation": PhysicsProfile("net_validation", 24000, 50, 64, 16, False, False),
}


@dataclass(frozen=True)
class LaunchSettings:
    speed_m_s: float = 22.0
    elevation_deg: float = 12.0
    azimuth_deg: float = 0.0
    side_spin_rps: float = 0.0
    top_spin_rps: float = 0.0
    rifle_spin_rps: float = 0.0
    launch_height_m: float = 0.25

    def __post_init__(self) -> None:
        if self.speed_m_s < 0.0:
            raise ValueError("launch speed cannot be negative")
        if not -89.0 < self.elevation_deg < 89.0:
            raise ValueError("elevation must be between -89 and 89 degrees")
        if self.launch_height_m < 0.0:
            raise ValueError("launch height cannot be negative")

    def velocity_world_m_s(self) -> tuple[float, float, float]:
        elevation = math.radians(self.elevation_deg)
        azimuth = math.radians(self.azimuth_deg)
        horizontal = self.speed_m_s * math.cos(elevation)
        return (
            horizontal * math.cos(azimuth),
            horizontal * math.sin(azimuth),
            self.speed_m_s * math.sin(elevation),
        )

    def angular_velocity_world_rad_s(self) -> tuple[float, float, float]:
        # Coordinates: +X toward goal, +Y left, +Z up.  Positive side spin
        # curves toward +Y; positive top spin (+Y axis) bends downward.
        factor = 2.0 * math.pi
        return (
            factor * self.rifle_spin_rps,
            factor * self.top_spin_rps,
            factor * self.side_spin_rps,
        )


FLIGHT_PRESETS: Mapping[str, LaunchSettings] = {
    "driven": LaunchSettings(speed_m_s=22.0, elevation_deg=12.0, launch_height_m=0.25),
    "curl": LaunchSettings(
        speed_m_s=24.0, elevation_deg=14.0, side_spin_rps=4.0, launch_height_m=0.25
    ),
    "knuckle": LaunchSettings(
        speed_m_s=25.0, elevation_deg=11.0, side_spin_rps=0.12, launch_height_m=0.25
    ),
}


def evidence_ledger() -> dict[str, dict[str, object]]:
    return {name: asdict(entry) for name, entry in PARAMETERS.items()}


def write_evidence_ledger(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence_ledger(), indent=2) + "\n", encoding="utf-8")
    return destination

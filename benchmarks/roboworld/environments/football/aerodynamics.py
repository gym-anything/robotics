"""Stateful six-degree-of-freedom football aerodynamics.

Native rigid-body simulation should own pose, velocity, gravity, and contact.
This module owns only the atmosphere-dependent wrench: buoyancy, drag,
Magnus/reverse-Magnus force, coherent low-spin wake force, and spin decay.

The coefficient values are transparent literature-scale priors.  The model is
structured as a replaceable interface so measured maps or a reduced CFD model
can be installed without changing the task, renderer, or Isaac adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Protocol

from .atmosphere import AirProperties, buoyancy_force_n
from .football_math import (
    IDENTITY_QUATERNION,
    Quat,
    Vec3,
    ZERO,
    add,
    clamp,
    cross,
    dot,
    norm,
    orthonormal_transverse_basis,
    quat_inverse_rotate,
    quat_rotate,
    reject,
    scale,
    unit,
)
from .football_specs import AerodynamicPrior, BallSpec
from .seam_geometry import SeamExposure, seam_exposure


@dataclass(frozen=True)
class BallKinematics:
    position_world_m: Vec3
    velocity_world_m_s: Vec3
    angular_velocity_world_rad_s: Vec3 = ZERO
    orientation_body_to_world: Quat = IDENTITY_QUATERNION


@dataclass(frozen=True)
class WakeState:
    amplitude: float = 0.0
    phase_rad: float = 0.0
    initialized: bool = False


@dataclass(frozen=True)
class AerodynamicCoefficients:
    drag: float
    magnus: float
    knuckle: float


@dataclass(frozen=True)
class ForceBreakdown:
    gravity_n: Vec3
    buoyancy_n: Vec3
    drag_n: Vec3
    magnus_n: Vec3
    knuckle_n: Vec3
    aerodynamic_torque_n_m: Vec3
    total_force_n: Vec3
    applied_force_without_engine_gravity_n: Vec3
    relative_velocity_world_m_s: Vec3
    speed_m_s: float
    reynolds_number: float
    spin_parameter: float
    dynamic_force_scale_n: float
    coefficients: AerodynamicCoefficients
    seam: SeamExposure
    wake: WakeState

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CoefficientModel(Protocol):
    def drag_coefficient(self, reynolds_number: float, spin_parameter: float, seam: SeamExposure) -> float:
        ...

    def magnus_coefficient(self, reynolds_number: float, spin_parameter: float) -> float:
        ...


@dataclass(frozen=True)
class PriorCoefficientModel:
    """Smooth generic-ball prior, explicitly not a branded-ball fit."""

    parameters: AerodynamicPrior = AerodynamicPrior()

    def drag_coefficient(self, reynolds_number: float, spin_parameter: float, seam: SeamExposure) -> float:
        p = self.parameters
        logistic = 1.0 / (1.0 + math.exp(
            clamp((reynolds_number - p.drag_critical_reynolds) / p.drag_transition_width, -60.0, 60.0)
        ))
        base = p.drag_supercritical + (p.drag_subcritical - p.drag_supercritical) * logistic
        spin_increment = p.drag_spin_gain * min(abs(spin_parameter), 0.5) ** 2
        # 0.88 is only a centering scale for the canonical seam geometry; the
        # modulation remains small compared with the drag-crisis change.
        seam_increment = p.drag_seam_modulation * (seam.projected_density - 0.88)
        return clamp(base + spin_increment + seam_increment, 0.08, 0.75)

    def magnus_coefficient(self, reynolds_number: float, spin_parameter: float) -> float:
        if spin_parameter <= 0.0:
            return 0.0
        p = self.parameters
        ordinary = min(p.magnus_cap, p.magnus_slope * spin_parameter)
        reverse_window = math.exp(-0.5 * (
            (reynolds_number - p.reverse_magnus_reynolds) / p.reverse_magnus_reynolds_width
        ) ** 2) * math.exp(-0.5 * (
            (spin_parameter - p.reverse_magnus_spin) / p.reverse_magnus_spin_width
        ) ** 2)
        return clamp(ordinary - p.reverse_magnus_amplitude * reverse_window, -0.08, p.magnus_cap)


class AerodynamicModel:
    def __init__(
        self,
        ball: BallSpec = BallSpec(),
        parameters: AerodynamicPrior = AerodynamicPrior(),
        coefficient_model: CoefficientModel | None = None,
        gravity_m_s2: float = 9.80665,
    ) -> None:
        self.ball = ball
        self.parameters = parameters
        self.coefficient_model = coefficient_model or PriorCoefficientModel(parameters)
        self.gravity_m_s2 = gravity_m_s2

    def _updated_wake(
        self,
        current: WakeState,
        *,
        dt_s: float,
        speed_m_s: float,
        spin_parameter: float,
        reynolds_number: float,
        seam: SeamExposure,
        seam_bias_world: Vec3,
        velocity_direction: Vec3,
    ) -> WakeState:
        p = self.parameters
        side, up = orthonormal_transverse_basis(velocity_direction)
        if current.initialized:
            phase = current.phase_rad
        else:
            lateral = dot(seam_bias_world, side)
            vertical = dot(seam_bias_world, up)
            # At an exact symmetry orientation the seam vector can vanish.
            # The density term still creates one deterministic (not random and
            # not target-selected) phase for that physical orientation.
            phase = (
                math.atan2(lateral, vertical)
                if abs(lateral) + abs(vertical) > 1.0e-12
                else 2.0 * math.pi * ((seam.projected_density * 1.61803398875) % 1.0)
            )

        low_spin_gate = math.exp(-(spin_parameter / p.knuckle_spin_scale) ** 2)
        speed_gate = 1.0 - math.exp(-(speed_m_s / 8.0) ** 2)
        crisis_gate = 0.35 + 0.65 * math.exp(-0.5 * (
            (reynolds_number - p.drag_critical_reynolds) / (1.8 * p.drag_transition_width)
        ) ** 2)
        orientation_gate = 0.35 + 0.65 * seam.asymmetry_strength
        target = clamp(low_spin_gate * speed_gate * crisis_gate * orientation_gate, 0.0, 1.0)
        if dt_s > 0.0:
            relaxation = 1.0 - math.exp(-dt_s / p.wake_relaxation_time_s)
            amplitude = current.amplitude + relaxation * (target - current.amplitude)
            frequency_hz = p.knuckle_reference_frequency_hz * speed_m_s / p.knuckle_reference_speed_m_s
            phase += 2.0 * math.pi * frequency_hz * dt_s
        else:
            amplitude = current.amplitude
        return WakeState(clamp(amplitude, 0.0, 1.0), phase % (2.0 * math.pi), True)

    def evaluate(
        self,
        kinematics: BallKinematics,
        air: AirProperties,
        wake: WakeState = WakeState(),
        dt_s: float = 0.0,
    ) -> ForceBreakdown:
        if dt_s < 0.0:
            raise ValueError("dt_s cannot be negative")

        ball = self.ball
        relative_velocity = air.relative_velocity(kinematics.velocity_world_m_s)
        speed = norm(relative_velocity)
        gravity = (0.0, 0.0, -ball.mass_kg * self.gravity_m_s2)
        buoyancy = (0.0, 0.0, buoyancy_force_n(ball, air, self.gravity_m_s2))
        empty_seam = seam_exposure((1.0, 0.0, 0.0))

        if speed < self.parameters.minimum_aero_speed_m_s:
            decayed = WakeState(
                wake.amplitude * math.exp(-dt_s / self.parameters.wake_relaxation_time_s) if dt_s else wake.amplitude,
                wake.phase_rad,
                wake.initialized,
            )
            total = add(gravity, buoyancy)
            return ForceBreakdown(
                gravity, buoyancy, ZERO, ZERO, ZERO, ZERO, total, buoyancy,
                relative_velocity, speed, 0.0, 0.0, 0.0,
                AerodynamicCoefficients(0.0, 0.0, 0.0), empty_seam, decayed,
            )

        velocity_direction = unit(relative_velocity)
        relative_body = quat_inverse_rotate(kinematics.orientation_body_to_world, relative_velocity)
        seam = seam_exposure(relative_body)
        seam_bias_world = quat_rotate(kinematics.orientation_body_to_world, seam.bias_body)
        reynolds_number = (
            air.density_kg_m3 * speed * ball.diameter_m / air.dynamic_viscosity_pa_s
        )
        transverse_spin = norm(reject(kinematics.angular_velocity_world_rad_s, velocity_direction))
        spin_parameter = transverse_spin * ball.radius_m / speed
        force_scale = 0.5 * air.density_kg_m3 * speed**2 * ball.projected_area_m2

        drag_coefficient = self.coefficient_model.drag_coefficient(reynolds_number, spin_parameter, seam)
        drag = scale(-force_scale * drag_coefficient, velocity_direction)

        magnus_direction = unit(cross(kinematics.angular_velocity_world_rad_s, relative_velocity))
        magnus_coefficient = self.coefficient_model.magnus_coefficient(reynolds_number, spin_parameter)
        magnus = scale(force_scale * magnus_coefficient, magnus_direction)

        updated_wake = self._updated_wake(
            wake,
            dt_s=dt_s,
            speed_m_s=speed,
            spin_parameter=spin_parameter,
            reynolds_number=reynolds_number,
            seam=seam,
            seam_bias_world=seam_bias_world,
            velocity_direction=velocity_direction,
        )
        side, up = orthonormal_transverse_basis(velocity_direction)
        knuckle_coefficient = self.parameters.knuckle_coefficient_cap * updated_wake.amplitude
        knuckle_direction = add(
            scale(math.sin(updated_wake.phase_rad), side),
            scale(self.parameters.wake_vertical_fraction * math.cos(updated_wake.phase_rad), up),
        )
        knuckle = scale(force_scale * knuckle_coefficient, knuckle_direction)

        angular_acceleration = scale(
            -self.parameters.spin_decay_per_m * speed,
            kinematics.angular_velocity_world_rad_s,
        )
        torque = scale(ball.thin_shell_inertia_kg_m2, angular_acceleration)
        applied_without_gravity = add(buoyancy, add(drag, add(magnus, knuckle)))
        total = add(gravity, applied_without_gravity)
        return ForceBreakdown(
            gravity_n=gravity,
            buoyancy_n=buoyancy,
            drag_n=drag,
            magnus_n=magnus,
            knuckle_n=knuckle,
            aerodynamic_torque_n_m=torque,
            total_force_n=total,
            applied_force_without_engine_gravity_n=applied_without_gravity,
            relative_velocity_world_m_s=relative_velocity,
            speed_m_s=speed,
            reynolds_number=reynolds_number,
            spin_parameter=spin_parameter,
            dynamic_force_scale_n=force_scale,
            coefficients=AerodynamicCoefficients(drag_coefficient, magnus_coefficient, knuckle_coefficient),
            seam=seam,
            wake=updated_wake,
        )

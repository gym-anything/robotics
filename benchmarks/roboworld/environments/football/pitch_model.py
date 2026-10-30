"""Measured-coupon-shaped pitch contact priors for a regulation football.

This is a force model, not a bounce script.  It returns a normal/tangential
wrench from instantaneous indentation, relative patch velocity, and contact
history.  PhysX (or the dependency-free coupon integrator) owns the resulting
rigid state.  The initial numbers target the published FIFA natural-pitch
performance bands and remain explicitly uncalibrated until a named pitch is
measured.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .football_math import Vec3, ZERO, add, cross, dot, norm, reject, scale, sub, unit
from .football_specs import BallSpec


@dataclass(frozen=True)
class PitchPrior:
    surface_family: str = "FIFA natural-pitch excellent-band initialization"
    evidence_status: str = "STANDARD-BAND-SHAPED PRIOR / NOT PITCH-CALIBRATED"
    normal_exponent: float = 1.50
    normal_stiffness_n_per_m_pow_n: float = 250_000.0
    normal_damping_ratio: float = 0.26
    damping_activation_depth_m: float = 0.025
    static_friction: float = 0.58
    dynamic_friction: float = 0.44
    tangential_stiffness_n_m: float = 14_000.0
    tangential_damping_n_s_m: float = 5.0
    # Effective rolling-resistance moment coefficient for the complete
    # grass/root-zone system.  Initialised against the FIFA 1.000 m roll-ramp
    # coupon, not interpreted as a blade-to-panel Coulomb coefficient.
    rolling_resistance_coefficient: float = 0.130
    spin_resistance_coefficient: float = 0.010
    pile_direction_world: Vec3 = (1.0, 0.0, 0.0)
    pile_friction_anisotropy: float = 0.08
    reference_drop_height_m: float = 2.0
    target_rebound_height_m: float = 0.80

    def __post_init__(self) -> None:
        if self.normal_exponent <= 1.0 or self.normal_stiffness_n_per_m_pow_n <= 0.0:
            raise ValueError("pitch normal law must be positive and superlinear")
        if self.normal_damping_ratio < 0.0 or self.damping_activation_depth_m <= 0.0:
            raise ValueError("pitch damping parameters are invalid")
        if not 0.0 <= self.dynamic_friction <= self.static_friction:
            raise ValueError("pitch friction must satisfy 0 <= dynamic <= static")
        if self.rolling_resistance_coefficient < 0.0 or self.spin_resistance_coefficient < 0.0:
            raise ValueError("pitch resistance coefficients must be non-negative")
        if norm(self.pile_direction_world) <= 0.0:
            raise ValueError("pile direction must be non-zero")


@dataclass(frozen=True)
class PitchContactState:
    tangential_displacement_m: Vec3 = ZERO
    in_contact: bool = False
    sliding: bool = False


@dataclass(frozen=True)
class PitchNormalForce:
    compression_m: float
    compression_rate_m_s: float
    elastic_n: float
    damping_n: float
    total_n: float
    elastic_energy_j: float
    damping_power_w: float


@dataclass(frozen=True)
class PitchWrench:
    force_on_ball_world_n: Vec3
    torque_on_ball_world_n_m: Vec3
    force_on_pitch_world_n: Vec3
    torque_on_pitch_world_n_m: Vec3
    tangential_force_on_ball_world_n: Vec3
    rolling_torque_on_ball_world_n_m: Vec3
    normal: PitchNormalForce
    state: PitchContactState


class PitchContactModel:
    def __init__(
        self,
        ball: BallSpec = BallSpec(),
        prior: PitchPrior = PitchPrior(),
    ) -> None:
        self.ball = ball
        self.prior = prior

    def normal_force(self, compression_m: float, compression_rate_m_s: float) -> PitchNormalForce:
        if compression_m <= 0.0:
            return PitchNormalForce(0.0, compression_rate_m_s, 0.0, 0.0, 0.0, 0.0, 0.0)
        p = self.prior
        delta = compression_m
        elastic = p.normal_stiffness_n_per_m_pow_n * delta**p.normal_exponent
        tangent_stiffness = (
            p.normal_exponent * p.normal_stiffness_n_per_m_pow_n
            * delta ** (p.normal_exponent - 1.0)
        )
        activation = min(1.0, delta / p.damping_activation_depth_m) ** 0.5
        damping_coefficient = (
            2.0 * p.normal_damping_ratio
            * math.sqrt(max(1.0, tangent_stiffness) * self.ball.mass_kg)
            * activation
        )
        trial_damping = damping_coefficient * compression_rate_m_s
        total = max(0.0, elastic + trial_damping)
        damping = total - elastic
        return PitchNormalForce(
            delta,
            compression_rate_m_s,
            elastic,
            damping,
            total,
            p.normal_stiffness_n_per_m_pow_n
            * delta ** (p.normal_exponent + 1.0)
            / (p.normal_exponent + 1.0),
            max(0.0, damping * compression_rate_m_s),
        )

    def ball_wrench(
        self,
        *,
        compression_m: float,
        ball_center_world_m: Vec3,
        ball_velocity_world_m_s: Vec3,
        ball_angular_velocity_world_rad_s: Vec3,
        contact_point_world_m: Vec3,
        normal_pitch_to_ball_world: Vec3,
        dt_s: float,
        state: PitchContactState = PitchContactState(),
    ) -> PitchWrench:
        if dt_s <= 0.0:
            raise ValueError("contact timestep must be positive")
        normal = unit(normal_pitch_to_ball_world)
        if norm(normal) <= 0.0:
            raise ValueError("pitch normal must be non-zero")
        arm = sub(contact_point_world_m, ball_center_world_m)
        ball_patch_velocity = add(
            ball_velocity_world_m_s,
            cross(ball_angular_velocity_world_rad_s, arm),
        )
        relative_pitch_minus_ball = scale(-1.0, ball_patch_velocity)
        compression_rate = dot(relative_pitch_minus_ball, normal)
        normal_result = self.normal_force(compression_m, compression_rate)
        if normal_result.total_n <= 0.0:
            return PitchWrench(
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, normal_result, PitchContactState()
            )

        tangent_velocity = reject(relative_pitch_minus_ball, normal)
        displacement = reject(
            add(state.tangential_displacement_m, scale(dt_s, tangent_velocity)), normal
        )
        trial = add(
            scale(self.prior.tangential_stiffness_n_m, displacement),
            scale(self.prior.tangential_damping_n_s_m, tangent_velocity),
        )
        # Pile anisotropy changes traction smoothly by direction, but the value
        # is a declared prior and never random per contact.
        pile = unit(reject(self.prior.pile_direction_world, normal), (1.0, 0.0, 0.0))
        along_fraction = 0.0 if norm(trial) <= 1.0e-12 else abs(dot(unit(trial), pile))
        friction_scale = 1.0 - self.prior.pile_friction_anisotropy * along_fraction
        static_limit = friction_scale * self.prior.static_friction * normal_result.total_n
        sliding = norm(trial) > static_limit
        if sliding:
            direction = unit(trial, unit(tangent_velocity))
            tangent_force = scale(
                friction_scale * self.prior.dynamic_friction * normal_result.total_n,
                direction,
            )
            displacement = scale(
                1.0 / self.prior.tangential_stiffness_n_m,
                sub(tangent_force, scale(self.prior.tangential_damping_n_s_m, tangent_velocity)),
            )
        else:
            tangent_force = trial

        normal_force = scale(normal_result.total_n, normal)
        force_on_ball = add(normal_force, tangent_force)
        contact_torque = cross(arm, force_on_ball)
        omega_roll = reject(ball_angular_velocity_world_rad_s, normal)
        omega_spin = scale(dot(ball_angular_velocity_world_rad_s, normal), normal)
        rolling_torque = add(
            scale(
                -self.prior.rolling_resistance_coefficient
                * normal_result.total_n * self.ball.radius_m,
                unit(omega_roll),
            ),
            scale(
                -self.prior.spin_resistance_coefficient
                * normal_result.total_n * self.ball.radius_m,
                unit(omega_spin),
            ),
        )
        torque_on_ball = add(contact_torque, rolling_torque)
        force_on_pitch = scale(-1.0, force_on_ball)
        # Static-pitch reaction is reported about the world origin.  A dynamic
        # pitch adapter may shift this to its own body centre.
        torque_on_pitch = scale(
            -1.0,
            add(cross(contact_point_world_m, force_on_ball), rolling_torque),
        )
        return PitchWrench(
            force_on_ball,
            torque_on_ball,
            force_on_pitch,
            torque_on_pitch,
            tangent_force,
            rolling_torque,
            normal_result,
            PitchContactState(displacement, True, sliding),
        )

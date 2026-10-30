"""Reduced-order pressurised-football contact model.

The model returns tractions/wrenches; it never chooses an outgoing ball state.
Its normal law separates pressure-volume work, nonlinear shell tension, and
rate hysteresis.  Tangential contact stores an explicit displacement history
and transitions from stick to a Coulomb-limited sliding state.

Parameters are literature-scale priors until fitted to a measured specimen and
held-out impact coupons.  SI units and force-on-ball sign conventions are used
throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .football_math import Vec3, ZERO, add, cross, dot, norm, reject, scale, sub, unit
from .football_specs import BallSpec


@dataclass(frozen=True)
class PressureShellPrior:
    initial_gauge_pressure_pa: float = 80_000.0
    atmospheric_pressure_pa: float = 101_325.0
    polytropic_exponent: float = 1.40
    volume_coupling: float = 0.22
    shell_exponent: float = 2.30
    reference_compression_m: float = 0.062
    reference_peak_force_n: float = 2926.0
    damping_ratio_prior: float = 0.20
    damping_activation_exponent: float = 0.50
    tangential_stiffness_n_m: float = 18_000.0
    tangential_damping_n_s_m: float = 7.0
    static_friction: float = 0.62
    dynamic_friction: float = 0.48

    def __post_init__(self) -> None:
        if self.initial_gauge_pressure_pa <= 0.0 or self.atmospheric_pressure_pa <= 0.0:
            raise ValueError("pressures must be positive")
        if self.polytropic_exponent <= 1.0:
            raise ValueError("polytropic exponent must exceed one")
        if not 0.0 < self.volume_coupling <= 1.0:
            raise ValueError("volume coupling must be in (0, 1]")
        if self.shell_exponent <= 1.0 or self.reference_compression_m <= 0.0:
            raise ValueError("shell exponent/compression are invalid")
        if self.reference_peak_force_n <= 0.0 or self.damping_ratio_prior < 0.0:
            raise ValueError("force must be positive and damping non-negative")
        if not 0.0 <= self.dynamic_friction <= self.static_friction:
            raise ValueError("friction must satisfy 0 <= dynamic <= static")


@dataclass(frozen=True)
class NormalForce:
    compression_m: float
    compression_rate_m_s: float
    total_n: float
    pressure_n: float
    shell_n: float
    damping_n: float
    internal_absolute_pressure_pa: float
    effective_volume_loss_m3: float
    shell_energy_j: float
    damping_power_w: float


@dataclass(frozen=True)
class ContactState:
    tangential_displacement_m: Vec3 = ZERO
    in_contact: bool = False
    sliding: bool = False


@dataclass(frozen=True)
class ContactWrench:
    force_on_ball_world_n: Vec3
    torque_on_ball_world_n_m: Vec3
    force_on_other_world_n: Vec3
    torque_on_other_world_n_m: Vec3
    normal: NormalForce
    tangential_force_on_ball_world_n: Vec3
    contact_point_world_m: Vec3
    state: ContactState


class PressureShellContactModel:
    """Stateful distributed-contact surrogate with a single patch resultant."""

    def __init__(
        self,
        ball: BallSpec = BallSpec(),
        prior: PressureShellPrior = PressureShellPrior(),
    ) -> None:
        self.ball = ball
        self.prior = prior
        reference_pressure = self.pressure_force(prior.reference_compression_m)[0]
        remaining = prior.reference_peak_force_n - reference_pressure
        if remaining <= 0.0:
            raise ValueError("pressure prior alone exceeds the reference peak force")
        self.shell_stiffness = remaining / prior.reference_compression_m**prior.shell_exponent

    @property
    def initial_absolute_pressure_pa(self) -> float:
        return self.prior.atmospheric_pressure_pa + self.prior.initial_gauge_pressure_pa

    def spherical_cap_volume_m3(self, compression_m: float) -> float:
        delta = max(0.0, min(compression_m, 2.0 * self.ball.radius_m))
        return math.pi * delta * delta * (self.ball.radius_m - delta / 3.0)

    def spherical_cap_area_derivative_m2(self, compression_m: float) -> float:
        delta = max(0.0, min(compression_m, 2.0 * self.ball.radius_m))
        return math.pi * delta * (2.0 * self.ball.radius_m - delta)

    def pressure_force(self, compression_m: float) -> tuple[float, float, float]:
        if compression_m <= 0.0:
            return 0.0, self.initial_absolute_pressure_pa, 0.0
        effective_loss = self.prior.volume_coupling * self.spherical_cap_volume_m3(compression_m)
        minimum_volume = 0.20 * self.ball.displaced_volume_m3
        remaining_volume = max(minimum_volume, self.ball.displaced_volume_m3 - effective_loss)
        ratio = self.ball.displaced_volume_m3 / remaining_volume
        absolute_pressure = self.initial_absolute_pressure_pa * ratio**self.prior.polytropic_exponent
        gauge_pressure = max(0.0, absolute_pressure - self.prior.atmospheric_pressure_pa)
        effective_area = (
            self.prior.volume_coupling * self.spherical_cap_area_derivative_m2(compression_m)
        )
        return gauge_pressure * effective_area, absolute_pressure, effective_loss

    def pressure_energy_j(self, compression_m: float) -> float:
        """Recoverable net gas/ambient work for the effective volume mode."""

        if compression_m <= 0.0:
            return 0.0
        _, _, effective_loss = self.pressure_force(compression_m)
        initial_volume = self.ball.displaced_volume_m3
        remaining_volume = max(0.20 * initial_volume, initial_volume - effective_loss)
        gamma = self.prior.polytropic_exponent
        gas_work = (
            self.initial_absolute_pressure_pa * initial_volume / (gamma - 1.0)
            * ((initial_volume / remaining_volume) ** (gamma - 1.0) - 1.0)
        )
        return gas_work - self.prior.atmospheric_pressure_pa * effective_loss

    def shell_energy_j(self, compression_m: float) -> float:
        delta = max(0.0, compression_m)
        return (
            self.shell_stiffness * delta ** (self.prior.shell_exponent + 1.0)
            / (self.prior.shell_exponent + 1.0)
        )

    def elastic_normal_force(self, compression_m: float) -> tuple[float, float, float, float]:
        if compression_m <= 0.0:
            return 0.0, 0.0, self.initial_absolute_pressure_pa, 0.0
        pressure, absolute_pressure, volume_loss = self.pressure_force(compression_m)
        shell = self.shell_stiffness * compression_m**self.prior.shell_exponent
        return pressure + shell, shell, absolute_pressure, volume_loss

    def tangent_stiffness_n_m(self, compression_m: float) -> float:
        delta = max(1.0e-7, compression_m)
        epsilon = max(1.0e-7, 1.0e-4 * delta)
        lower = max(0.0, delta - epsilon)
        upper = delta + epsilon
        lower_force = self.elastic_normal_force(lower)[0]
        upper_force = self.elastic_normal_force(upper)[0]
        return max(1.0, (upper_force - lower_force) / (upper - lower))

    def normal_force(
        self,
        compression_m: float,
        compression_rate_m_s: float,
        reduced_mass_kg: float,
    ) -> NormalForce:
        if reduced_mass_kg <= 0.0:
            raise ValueError("reduced mass must be positive")
        if compression_m <= 0.0:
            return NormalForce(
                0.0, compression_rate_m_s, 0.0, 0.0, 0.0, 0.0,
                self.initial_absolute_pressure_pa, 0.0, 0.0, 0.0,
            )
        pressure, absolute_pressure, volume_loss = self.pressure_force(compression_m)
        shell = self.shell_stiffness * compression_m**self.prior.shell_exponent
        activation = min(1.0, compression_m / self.prior.reference_compression_m)
        activation **= self.prior.damping_activation_exponent
        damping_coefficient = (
            2.0 * self.prior.damping_ratio_prior
            * math.sqrt(self.tangent_stiffness_n_m(compression_m) * reduced_mass_kg)
            * activation
        )
        damping = damping_coefficient * compression_rate_m_s
        total = max(0.0, pressure + shell + damping)
        # If tensile damping would exceed the elastic force, the contact opens;
        # report the actually applied damping rather than the unclipped trial.
        applied_damping = total - pressure - shell
        shell_energy = self.shell_energy_j(compression_m)
        damping_power = max(0.0, applied_damping * compression_rate_m_s)
        return NormalForce(
            compression_m,
            compression_rate_m_s,
            total,
            pressure,
            shell,
            applied_damping,
            absolute_pressure,
            volume_loss,
            shell_energy,
            damping_power,
        )

    def contact_wrench(
        self,
        *,
        compression_m: float,
        relative_velocity_other_minus_ball_world_m_s: Vec3,
        normal_other_to_ball_world: Vec3,
        contact_point_world_m: Vec3,
        ball_center_world_m: Vec3,
        other_center_world_m: Vec3,
        reduced_mass_kg: float,
        dt_s: float,
        state: ContactState = ContactState(),
    ) -> ContactWrench:
        if dt_s <= 0.0:
            raise ValueError("contact timestep must be positive")
        normal = unit(normal_other_to_ball_world)
        if norm(normal) <= 0.0:
            raise ValueError("contact normal must be non-zero")
        relative_velocity = relative_velocity_other_minus_ball_world_m_s
        compression_rate = dot(relative_velocity, normal)
        normal_result = self.normal_force(compression_m, compression_rate, reduced_mass_kg)
        if normal_result.total_n <= 0.0:
            separated = ContactState()
            return ContactWrench(ZERO, ZERO, ZERO, ZERO, normal_result, ZERO,
                                 contact_point_world_m, separated)

        tangent_velocity = reject(relative_velocity, normal)
        old_displacement = reject(state.tangential_displacement_m, normal)
        displacement = reject(
            add(old_displacement, scale(dt_s, tangent_velocity)), normal
        )
        midpoint_displacement = scale(
            0.5,
            add(old_displacement, displacement),
        )
        trial_tangent = add(
            scale(self.prior.tangential_stiffness_n_m, midpoint_displacement),
            scale(self.prior.tangential_damping_n_s_m, tangent_velocity),
        )
        static_limit = self.prior.static_friction * normal_result.total_n
        sliding = norm(trial_tangent) > static_limit
        if sliding:
            # Coulomb slip has no recoverable bristle memory in this reduced
            # model.  Resetting the tangential spring makes the slip branch
            # discretely passive; carrying a force-clamped spring displacement
            # as before could create energy as the normal load changed.
            direction = unit(tangent_velocity, unit(trial_tangent))
            tangent_force = scale(self.prior.dynamic_friction * normal_result.total_n, direction)
            displacement = ZERO
        else:
            tangent_force = trial_tangent

        force_on_ball = add(scale(normal_result.total_n, normal), tangent_force)
        force_on_other = scale(-1.0, force_on_ball)
        torque_on_ball = cross(sub(contact_point_world_m, ball_center_world_m), force_on_ball)
        torque_on_other = cross(sub(contact_point_world_m, other_center_world_m), force_on_other)
        next_state = ContactState(displacement, True, sliding)
        return ContactWrench(
            force_on_ball,
            torque_on_ball,
            force_on_other,
            torque_on_other,
            normal_result,
            tangent_force,
            contact_point_world_m,
            next_state,
        )

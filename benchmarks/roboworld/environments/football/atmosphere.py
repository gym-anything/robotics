"""Moist-air properties for football flight.

Density follows the CIPM-2007 moist-air formulation rather than a fixed
``1.225 kg/m^3`` constant.  Dynamic viscosity uses Sutherland's law.  This is
still an atmosphere model, not a weather simulator: spatially varying gusts
can later implement the same ``Atmosphere`` interface without changing the
ball or Isaac adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .football_math import Vec3, sub
from .football_specs import AtmosphereSpec, BallSpec


@dataclass(frozen=True)
class AirProperties:
    density_kg_m3: float
    dynamic_viscosity_pa_s: float
    vapor_mole_fraction: float
    compressibility: float
    temperature_k: float
    pressure_pa: float
    wind_velocity_world_m_s: Vec3

    def relative_velocity(self, body_velocity_world_m_s: Vec3) -> Vec3:
        """Velocity of the body relative to the local air."""

        return sub(body_velocity_world_m_s, self.wind_velocity_world_m_s)


def saturation_vapor_pressure_pa(temperature_k: float) -> float:
    """CIPM water-vapor saturation pressure for ordinary ambient conditions."""

    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    a = 1.2378847e-5
    b = -1.9121316e-2
    c = 33.93711047
    d = -6.3431645e3
    return math.exp(a * temperature_k**2 + b * temperature_k + c + d / temperature_k)


def enhancement_factor(temperature_k: float, pressure_pa: float) -> float:
    temperature_c = temperature_k - 273.15
    return 1.00062 + 3.14e-8 * pressure_pa + 5.6e-7 * temperature_c**2


def vapor_mole_fraction(spec: AtmosphereSpec) -> float:
    value = (
        spec.relative_humidity
        * enhancement_factor(spec.temperature_k, spec.pressure_pa)
        * saturation_vapor_pressure_pa(spec.temperature_k)
        / spec.pressure_pa
    )
    return min(1.0, max(0.0, value))


def compressibility_factor(spec: AtmosphereSpec, mole_fraction: float | None = None) -> float:
    """CIPM-2007 compressibility factor Z for moist air."""

    t_c = spec.temperature_k - 273.15
    x_v = vapor_mole_fraction(spec) if mole_fraction is None else mole_fraction
    a0, a1, a2 = 1.58123e-6, -2.9331e-8, 1.1043e-10
    b0, b1 = 5.707e-6, -2.051e-8
    c0, c1 = 1.9898e-4, -2.376e-6
    d, e = 1.83e-11, -0.765e-8
    p_over_t = spec.pressure_pa / spec.temperature_k
    first = a0 + a1 * t_c + a2 * t_c**2 + (b0 + b1 * t_c) * x_v + (c0 + c1 * t_c) * x_v**2
    second = d + e * x_v**2
    return 1.0 - p_over_t * first + p_over_t**2 * second


def moist_air_density_kg_m3(spec: AtmosphereSpec) -> float:
    x_v = vapor_mole_fraction(spec)
    z = compressibility_factor(spec, x_v)
    molar_mass_dry_air = 28.96546e-3
    molar_mass_water = 18.01528e-3
    molar_gas_constant = 8.314472
    return (
        spec.pressure_pa
        * molar_mass_dry_air
        / (z * molar_gas_constant * spec.temperature_k)
        * (1.0 - x_v * (1.0 - molar_mass_water / molar_mass_dry_air))
    )


def dynamic_viscosity_pa_s(temperature_k: float) -> float:
    """Sutherland viscosity prior for air."""

    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    reference_temperature_k = 273.15
    reference_viscosity_pa_s = 1.716e-5
    sutherland_temperature_k = 111.0
    return (
        reference_viscosity_pa_s
        * (temperature_k / reference_temperature_k) ** 1.5
        * (reference_temperature_k + sutherland_temperature_k)
        / (temperature_k + sutherland_temperature_k)
    )


def evaluate_atmosphere(spec: AtmosphereSpec = AtmosphereSpec()) -> AirProperties:
    x_v = vapor_mole_fraction(spec)
    return AirProperties(
        density_kg_m3=moist_air_density_kg_m3(spec),
        dynamic_viscosity_pa_s=dynamic_viscosity_pa_s(spec.temperature_k),
        vapor_mole_fraction=x_v,
        compressibility=compressibility_factor(spec, x_v),
        temperature_k=spec.temperature_k,
        pressure_pa=spec.pressure_pa,
        wind_velocity_world_m_s=spec.wind_velocity_world_m_s,
    )


def buoyancy_force_n(ball: BallSpec, air: AirProperties, gravity_m_s2: float = 9.80665) -> float:
    return air.density_kg_m3 * ball.displaced_volume_m3 * gravity_m_s2

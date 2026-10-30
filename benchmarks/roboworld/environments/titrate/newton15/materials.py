"""SI water reference properties and explicit numerical configuration.

Water near 20 C is the reference carrier for dilute titration solutions.
These are literature values, not specimen measurements.
"""

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class Water:
    temperature_k: float = 293.15
    density_kg_m3: float = 998.2
    viscosity_pa_s: float = 0.001002

    @property
    def surface_tension_n_m(self):
        # IAPWS R1-76(2014), equation 1; critical temperature 647.096 K.
        tau = 1.0 - self.temperature_k / 647.096
        if not 273.16 <= self.temperature_k < 647.096:
            raise ValueError("Temperature outside the IAPWS saturation-curve range")
        return 0.2358 * tau**1.256 * (1.0 - 0.625 * tau)

    def __post_init__(self):
        for value in asdict(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Material properties must be finite and positive")
        if self.temperature_k != 293.15:
            raise ValueError("Density and viscosity are currently specified only at 293.15 K")


@dataclass(frozen=True)
class FluidNumerics:
    voxel_m: float = 0.001
    particles_per_cell_axis: int = 2
    dt_s: float = 0.0005
    max_iterations: int = 250
    tolerance: float = 1.0e-6
    device: str = "cuda:0"
    capillarity: bool = False
    velocity_basis: str = "Q1"
    strain_basis: str = "P0"
    surface_model: str = "balanced"
    tensile_limit_pa: float = 0.0
    critical_fraction: float = 0.0
    air_drag: float = 0.0
    integration_scheme: str = "pic"
    verbose: bool = False
    bulk_modulus_pa: float | None = None

    @property
    def surface_support_m(self):
        return (4.0 if self.surface_model == "balanced" else 3.0) * self.spacing_m

    @property
    def spacing_m(self):
        return self.voxel_m / self.particles_per_cell_axis

    def __post_init__(self):
        for value in (self.voxel_m, self.dt_s, self.tolerance):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Fluid numerical scales must be finite and positive")
        for value in (self.particles_per_cell_axis, self.max_iterations):
            if type(value) is not int or value < 1:
                raise ValueError("Fluid counts must be positive integers")
        if type(self.capillarity) is not bool:
            raise ValueError("capillarity must be an explicit Boolean")
        if self.velocity_basis not in ("Q1", "B2", "B3") or self.strain_basis not in ("P0", "P1d", "pic"):
            raise ValueError("Unsupported explicit MPM basis selection")
        if self.surface_model not in ("balanced", "attraction_only", "area", "csf", "mesh", "interface_energy"):
            raise ValueError("Unknown capillary discretization")
        if not math.isfinite(self.tensile_limit_pa) or self.tensile_limit_pa < 0:
            raise ValueError("Pressure limit must be finite and nonnegative")
        for value in (self.critical_fraction, self.air_drag):
            if not math.isfinite(value) or value < 0:
                raise ValueError("MPM fraction and drag must be finite and nonnegative")
        if self.integration_scheme not in ("pic", "gimp"):
            raise ValueError("Unknown particle integration scheme")
        if self.bulk_modulus_pa is not None and (not math.isfinite(self.bulk_modulus_pa) or self.bulk_modulus_pa <= 0):
            raise ValueError("Finite positive bulk modulus or explicit incompressible limit required")

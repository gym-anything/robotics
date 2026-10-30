"""Lumped isothermal air chamber, independent of liquid/rigid implementation.

This component does NOT prescribe liquid flow or implement a suction boundary.
Coupling must supply the actual gas volume and apply pressure work to liquid
and piston surfaces. Constant chamber temperature is an explicit approximation;
the vent uses the compressible ideal-nozzle relation, including choking.
Basis: https://www.grc.nasa.gov/www/k-12/airplane/mflchk.html
"""

from dataclasses import dataclass
import math
from scipy.optimize import brentq


def finite(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Finite {name} required')
    if (zero and value < 0.) or (not zero and value <= 0.):
        raise ValueError(f'{name} must be {"nonnegative" if zero else "positive"}')
    return float(value)


@dataclass(frozen=True)
class Air:
    temperature_k: float = 293.15
    ambient_pressure_pa: float = 101325.
    specific_gas_constant_j_kg_k: float = 287.05
    heat_capacity_ratio: float = 1.4

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            finite(getattr(self, name), name)
        if self.heat_capacity_ratio <= 1.:
            raise ValueError('Heat-capacity ratio must exceed one')


def orifice_mass_flow(upstream_pa, downstream_pa, area_m2, discharge_coefficient, air):
    """Positive upstream-to-downstream kg/s; pressures are absolute.

    Area and discharge coefficient must be explicit caller-supplied geometry
    and loss assumptions, not a hidden desired pipetting speed.
    """
    pu = finite(upstream_pa, 'upstream pressure', zero=True)
    pd = finite(downstream_pa, 'downstream pressure', zero=True)
    area = finite(area_m2, 'vent area', zero=True)
    cd = finite(discharge_coefficient, 'discharge coefficient')
    if pd > pu or cd > 1.:
        raise ValueError('Ordered pressures and discharge coefficient in (0, 1] required')
    if pu == pd or area == 0.:
        return 0.
    gamma = air.heat_capacity_ratio
    rt = air.specific_gas_constant_j_kg_k*air.temperature_k
    ratio = pd/pu
    critical = (2./(gamma+1.))**(gamma/(gamma-1.))
    if ratio <= critical:
        flux = math.sqrt(gamma/rt)*(2./(gamma+1.))**((gamma+1.)/(2.*(gamma-1.)))
    else:
        # Stable near equal pressures: r^(2/gamma) * (1-r^((gamma-1)/gamma)).
        difference = -math.expm1((gamma-1.)/gamma*math.log(ratio))
        flux = math.sqrt(2.*gamma/(rt*(gamma-1.))*ratio**(2./gamma)*difference)
    return cd*area*pu*flux


class IsothermalChamber:
    def __init__(self, volume_m3, *, pressure_pa=None, air=None):
        self.air = air or Air()
        self.volume_m3 = finite(volume_m3, 'gas volume')
        pressure = self.air.ambient_pressure_pa if pressure_pa is None else finite(pressure_pa, 'pressure', zero=True)
        self.mass_kg = pressure*self.volume_m3/self.rt

    @property
    def rt(self):
        return self.air.specific_gas_constant_j_kg_k*self.air.temperature_k

    @property
    def pressure_pa(self):
        return self.mass_kg*self.rt/self.volume_m3

    def change_volume(self, volume_m3):
        """Sealed isothermal volume change; return exact gas and gauge work.

        Positive work is done BY gas during expansion. No liquid volume,
        particle pose or piston position is assigned by this method.
        """
        new = finite(volume_m3, 'gas volume')
        old = self.volume_m3
        work = self.mass_kg*self.rt*math.log(new/old)
        self.volume_m3 = new
        return dict(gas_work_j=work,
            gauge_work_j=work-self.air.ambient_pressure_pa*(new-old),
            pressure_pa=self.pressure_pa, mass_kg=self.mass_kg)

    def vent(self, dt_s, *, area_m2, discharge_coefficient):
        """Backward-Euler mass exchange at fixed volume, without overshoot.

        Ambient is the declared reservoir. Incoming/outgoing gas is recorded
        explicitly; chamber mass is not reset to atmospheric equilibrium.
        """
        dt = finite(dt_s, 'vent time')
        area = finite(area_m2, 'vent area', zero=True)
        cd = finite(discharge_coefficient, 'discharge coefficient')
        if cd > 1.:
            raise ValueError('Discharge coefficient must not exceed one')
        old, ambient = self.pressure_pa, self.air.ambient_pressure_pa
        def flow(p):
            if p <= ambient:
                return orifice_mass_flow(ambient, p, area, cd, self.air)
            return -orifice_mass_flow(p, ambient, area, cd, self.air)
        if old == ambient or area == 0.:
            return dict(mass_exchange_kg=0., pressure_pa=old, backward_euler_residual_kg=0.)
        # Dimensionless pressure keeps the root tolerance independent of units.
        scale = dt*self.rt/(self.volume_m3*ambient)
        ratio = brentq(lambda r: r-old/ambient-scale*flow(r*ambient),
            min(old/ambient, 1.), max(old/ambient, 1.), xtol=1.e-14, rtol=1.e-14)
        mass = ratio*ambient*self.volume_m3/self.rt
        exchange = mass-self.mass_kg
        residual = exchange-dt*flow(ratio*ambient)
        self.mass_kg = mass
        return dict(mass_exchange_kg=exchange, pressure_pa=self.pressure_pa,
            backward_euler_residual_kg=residual)

    def piston_force_n(self, area_m2):
        """Force in the volume-increasing direction, relative to ambient."""
        return (self.pressure_pa-self.air.ambient_pressure_pa)*finite(area_m2, 'piston area')

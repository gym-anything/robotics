"""Conservative pressure-port coupling of a lumped air chamber to SPH liquid.

The liquid is still resolved by the existing solver. Only the connected gas
is lumped. No particles, velocities or flow rates are assigned.
"""

import numpy as np
import warp as wp
from .pipette_geometry import profile, LENGTH_M, WALL_M, FACET_AREA_RATIO, capacity
from .pneumatics import IsothermalChamber


@wp.func
def step01(x: float):
    t = wp.clamp(x, 0., 1.)
    return t*t*(3.-2.*t)


@wp.func
def step01_derivative(x: float):
    d = float(0.)
    if x > 0. and x < 1.:
        d = 6.*x*(1.-x)
    return d


@wp.kernel
def cavity_fraction(q: wp.array[wp.vec3], pose: wp.transform,
                    meridian: wp.array[wp.vec2], port_width: float,
                    weights: wp.array[float], gradients: wp.array[wp.vec3]):
    i = wp.tid()
    p = wp.transform_point(wp.transform_inverse(pose), q[i])
    r = wp.sqrt(p[0]*p[0]+p[1]*p[1])
    radius, slope = meridian[0][0], float(0.)
    for k in range(meridian.shape[0]-1):
        a, b = meridian[k], meridian[k+1]
        if p[2] >= a[1]:
            t = wp.clamp((p[2]-a[1])/(b[1]-a[1]), 0., 1.)
            radius = a[0]+t*(b[0]-a[0])
            slope = 0.
            if p[2] < b[1]:
                slope = (b[0]-a[0])/(b[1]-a[1])
    # Radial transition lies inside the 0.8 mm solid glass wall. It is not
    # a changed collision wall or a force throughout the source vessel.
    radial_width = float(.0004)
    u = (radius+.0006-r)/radial_width
    radial = step01(u)
    dr = step01_derivative(u)/radial_width
    lower_arg = (p[2]+.5*port_width)/port_width
    upper_arg = (float(.429)+.5*port_width-p[2])/port_width
    lower, upper = step01(lower_arg), step01(upper_arg)
    dz = (step01_derivative(lower_arg)*upper-lower*step01_derivative(upper_arg))/port_width
    grad = wp.vec3(0., 0., radial*dz+dr*slope*lower*upper)
    if r > 0.:
        grad[0] = -dr*p[0]/r*lower*upper
        grad[1] = -dr*p[1]/r*lower*upper
    weights[i] = radial*lower*upper
    gradients[i] = wp.transform_vector(pose, grad)


@wp.kernel
def pressure_predict(q: wp.array[wp.vec3], gradients: wp.array[wp.vec3],
                     gauge_pa: float, parcel_volume: float, density: float, dt: float,
                     com: wp.vec3, velocity: wp.array[wp.vec3],
                     reaction: wp.array[wp.spatial_vector]):
    i = wp.tid()
    force = -gauge_pa*parcel_volume*gradients[i]
    # Most parcels are outside this tool. Do not serialize six zero-valued
    # atomic additions per parcel onto one reaction accumulator. This is an
    # exact zero test, not a force threshold or spatial approximation.
    if force[0] != 0. or force[1] != 0. or force[2] != 0.:
        velocity[i] += force*(dt/(density*parcel_volume))
        wp.atomic_add(reaction, 0, wp.spatial_vector(-force, wp.cross(q[i]-com, -force)))


class PipettePressure:
    def __init__(self, water, *, dead_volume_m3=2.e-6):
        self.water = water
        # Smooth over the kernel diameter. A one-spacing port can fall between
        # successive particle layers and numerically lose almost all pressure
        # traction even while a continuous column occupies the tip.
        self.port_width_m = 2.*water.config.support_m
        inner = np.asarray(profile())[7:][::-1].copy()
        self.capacity_m3 = FACET_AREA_RATIO*capacity(inner, LENGTH_M)
        self.dead_volume_m3 = float(dead_volume_m3)
        self.meridian = wp.array(inner, dtype=wp.vec2, device=water.config.device)
        self.weights = wp.zeros(water.model.particle_count, dtype=float, device=water.config.device)
        self.gradients = wp.zeros(water.model.particle_count, dtype=wp.vec3, device=water.config.device)
        self.reaction = wp.zeros(1, dtype=wp.spatial_vector, device=water.config.device)
        self.total = wp.zeros(1, dtype=float, device=water.config.device)
        self.gas = None
        self.last = {}
        self.pose = None

    def measure(self, pose):
        self.pose = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
        wp.launch(cavity_fraction, self.water.model.particle_count,
            inputs=[self.water.state.particle_q, self.pose, self.meridian,
                    self.port_width_m], outputs=[self.weights, self.gradients],
            device=self.water.config.device)
        wp.utils.array_sum(self.weights, out=self.total)
        return float(self.total.numpy()[0])*self.water.volume_per_particle_m3

    def prepare(self, pose, com, *, displaced_volume_m3, vent_area_m2=0., tip_air_area_m2=0.):
        occupied = self.measure(pose)
        volume = self.capacity_m3+self.dead_volume_m3+displaced_volume_m3-occupied
        if self.gas is None:
            self.gas = IsothermalChamber(volume)
        work = self.gas.change_volume(volume)
        vent = self.gas.vent(self.water.config.dt_s,
            area_m2=vent_area_m2+tip_air_area_m2, discharge_coefficient=.65)
        self.gauge_pa = self.gas.pressure_pa-self.gas.air.ambient_pressure_pa
        self.com = wp.vec3(*com)
        self.last = {**work, **vent, 'occupied_volume_ml': occupied*1.e6,
            'gas_volume_ml': volume*1.e6, 'gas_pressure_pa': self.gas.pressure_pa,
            'gauge_pressure_pa': self.gauge_pa,
            'pressure_port_width_m': self.port_width_m,
            'displaced_volume_ml': displaced_volume_m3*1.e6}

    def apply(self, water):
        if self.pose is None:
            raise RuntimeError('Pipette pressure must sample its physical pose before stepping')
        self.reaction.zero_()
        wp.launch(pressure_predict, water.model.particle_count,
            inputs=[water.state.particle_q, self.gradients, self.gauge_pa,
                water.volume_per_particle_m3, water.material.density_kg_m3,
                water.config.dt_s, self.com],
            outputs=[water.predicted_velocity, self.reaction], device=water.config.device)

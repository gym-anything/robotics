"""SI-normalized comparison using Akinci et al.'s published cohesion kernel.

This uses their signed cohesion kernel, NOT their complete cohesion+curvature
algorithm or dimensionless demo coefficient. Pair potential Phi'=C, Phi(h)=0.
For continuum density rho, one planar interface has energy
sigma = gamma*rho^2*pi/8 * integral_0^h r^4 C(r) dr.
The integral gives gamma=(7040/21)*sigma/(rho^2*h^2). Solid adhesion uses
the same potential with gamma_b=(1+cos(theta))*gamma/2, by Young's law.
No drop measurements or physical parameters are fitted to the outcome.

Kernel source: SPlisHSPlasH/SPHKernels.h, CohesionKernel; Akinci et al.,
ACM TOG 32(6), 2013, DOI 10.1145/2508363.2508395.
"""

import math
import warp as wp
from .capillary import CapillaryPredictor


@wp.func
def cohesion(x: float):
    value = float(0.)
    if x >= 0. and x <= 1.:
        value = wp.pow(1.-x, 3.)*x*x*x
        if x <= .5:
            value = 2.*value-1./64.
    return (32./wp.pi)*value


@wp.func
def primitive(x: float, k: int):
    return (wp.pow(x, float(k+4))/float(k+4)-3.*wp.pow(x, float(k+5))/float(k+5)
            +3.*wp.pow(x, float(k+6))/float(k+6)-wp.pow(x, float(k+7))/float(k+7))


@wp.func
def tail(s: float, k: int):
    result = float(0.)
    if s < 1.:
        result = primitive(1., k)-primitive(wp.max(s, .5), k)
        if s < .5:
            result += 2.*(primitive(.5, k)-primitive(s, k))
            result -= (wp.pow(.5, float(k+1))-wp.pow(s, float(k+1)))/(64.*float(k+1))
    return (32./wp.pi)*result


@wp.kernel
def predict(grid: wp.uint64, q: wp.array[wp.vec3], v: wp.array[wp.vec3], mass: wp.array[float],
            h: float, gamma: float, density: float, dt: float, wall: int, cosine: float,
            out: wp.array[wp.vec3], forces: wp.array[wp.vec3]):
    i = wp.tid()
    force = wp.vec3(0.)
    for j in wp.hash_grid_query(grid, q[i], h):
        d = q[i]-q[j]
        r = wp.length(d)
        if j != i and r > 1.e-12 and r < h:
            force -= gamma*mass[i]*mass[j]*cohesion(r/h)/(h*h*h)*d/r
    if wall != 0 and q[i][2] < h:
        s = wp.max(0., q[i][2]/h)
        integral = wp.pi*(tail(s, 2)-s*s*tail(s, 0))
        force[2] -= .5*(1.+cosine)*gamma*density*mass[i]*integral
    forces[i] = force
    out[i] = v[i]+dt*force/mass[i]


class CohesionPredictor(CapillaryPredictor):
    def predict(self, state, dt):
        sigma = self.sigma_n_m
        self.sigma_n_m = 0.
        try:
            predicted = super().predict(state, dt)
        finally:
            self.sigma_n_m = sigma
        rho = float(self.model.particle_mass.numpy()[0]/self.volumes.numpy()[0])
        gamma = (7040./21.)*sigma/(rho*rho*self.support_m**2)
        angle = getattr(self, "wall_angle", None)
        self.wp.launch(predict, dim=self.model.particle_count,
            inputs=[self.grid.id, state.particle_q, state.particle_qd, self.model.particle_mass,
                self.support_m, gamma, rho, dt, int(angle is not None), math.cos(math.radians(angle or 0.))],
            outputs=[predicted.particle_qd, self.forces], device=self.model.device)
        return predicted

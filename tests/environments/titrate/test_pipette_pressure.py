import unittest
from types import SimpleNamespace
import numpy as np
import warp as wp
from titrate.newton15.runtime import require_newton
from titrate.newton15.pipette_pressure import PipettePressure


class PressurePortTests(unittest.TestCase):
    def test_port_quadrature_does_not_vanish_between_layers(self):
        # Integral of d(chi)/dz along a filled lattice line should be one.
        # One-dx support can entirely miss the layer centers. Four-dx support
        # has a bounded 6.25% quadrature error for all tested axial phases.
        errors = []
        for phase in np.linspace(0., 1., 101):
            z = np.arange(-5., 6.)+phase
            u = (z+2.)/4.
            inside = (u > 0.) & (u < 1.)
            integral = (6.*u[inside]*(1.-u[inside])/4.).sum()
            errors.append(abs(integral-1.))
        self.assertLessEqual(max(errors), .0625001)

    def make(self):
        require_newton()
        q = np.array([[0., 0., .0001], [.02, 0., .0001], [0., 0., .010]], np.float32)
        water = SimpleNamespace(model=SimpleNamespace(particle_count=len(q)),
            config=SimpleNamespace(device='cpu', spacing_m=.000625, support_m=.00125, dt_s=.000125),
            material=SimpleNamespace(density_kg_m3=998.2), volume_per_particle_m3=.000625**3,
            state=SimpleNamespace(particle_q=wp.array(q, dtype=wp.vec3, device='cpu')),
            predicted_velocity=wp.zeros(len(q), dtype=wp.vec3, device='cpu'))
        return water, PipettePressure(water)

    def test_ambient_and_action_reaction(self):
        water, port = self.make()
        pose = np.array([0., 0., 0., 0., 0., 0., 1.])
        port.prepare(pose, [0., 0., 0.], displaced_volume_m3=0.)
        port.apply(water)
        np.testing.assert_allclose(water.predicted_velocity.numpy(), 0., atol=1.e-11)
        port.prepare(pose, [0., 0., 0.], displaced_volume_m3=.1e-6)
        port.apply(water)
        fluid = water.predicted_velocity.numpy().sum(axis=0)*998.2*water.volume_per_particle_m3/water.config.dt_s
        np.testing.assert_allclose(fluid+port.reaction.numpy()[0, :3], 0., atol=1.e-10)
        self.assertGreater(fluid[2], 0.)
        self.assertEqual(float(port.weights.numpy()[1]), 0.)  # exterior source liquid unaffected

    def test_force_is_gas_energy_gradient(self):
        water, port = self.make()
        pose = np.array([0., 0., 0., 0., 0., 0., 1.])
        port.prepare(pose, [0., 0., 0.], displaced_volume_m3=0.)
        displacement = .1e-6
        port.prepare(pose, [0., 0., 0.], displaced_volume_m3=displacement)
        force = -port.gauge_pa*water.volume_per_particle_m3*port.gradients.numpy()[0, 2]
        q = water.state.particle_q.numpy().copy()
        def energy(offset):
            moved = q.copy(); moved[0, 2] += offset
            water.state.particle_q.assign(moved)
            volume = port.capacity_m3+port.dead_volume_m3+displacement-port.measure(pose)
            # Relative energy avoids subtracting two ~1 J absolute values.
            reference = port.gas.volume_m3
            return -port.gas.mass_kg*port.gas.rt*np.log(volume/reference)+port.gas.air.ambient_pressure_pa*(volume-reference)
        delta = 2.e-6
        numeric = -(energy(delta)-energy(-delta))/(2.*delta)
        self.assertAlmostEqual(numeric/force, 1., delta=2.e-4)


if __name__ == '__main__':
    unittest.main()

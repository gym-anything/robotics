"""Analytical/discrete constitutive checks, not full liquid validation."""

import math
import unittest

import numpy as np
from scipy.integrate import quad

from titrate.newton15.capillary import kernel, reference
from titrate.newton15.materials import Water


class CapillaryTests(unittest.TestCase):
    def test_iapws_20_c(self):
        self.assertAlmostEqual(Water().surface_tension_n_m, .07274, delta=5e-6)

    def test_normalized_kernel_and_flat_surface_energy(self):
        for h in (.0005, .001, .002):
            norm = quad(lambda r: 4*math.pi*r*r*kernel(r, h), 0, h)[0]
            j = quad(lambda r: math.pi*r**3*kernel(r, h), 0, h)[0]
            self.assertAlmostEqual(norm, 1.0, places=12)
            self.assertAlmostEqual(j / h, 5.0/48.0, places=12)

    def test_force_is_negative_energy_gradient(self):
        self.check_gradient(0.0)
        self.check_gradient(.0008)

    def check_gradient(self, repulsive):
        q = np.random.default_rng(7).uniform(-.0008, .0008, (14, 3))
        volumes = np.full(len(q), (.0003)**3)
        sigma = Water().surface_tension_n_m
        _, f = reference(q, volumes, .0012, sigma, repulsive)
        finite = np.zeros_like(f)
        epsilon = 1e-8
        for i in range(len(q)):
            for axis in range(3):
                plus, minus = q.copy(), q.copy()
                plus[i, axis] += epsilon
                minus[i, axis] -= epsilon
                ep = reference(plus, volumes, .0012, sigma, repulsive)[0]
                em = reference(minus, volumes, .0012, sigma, repulsive)[0]
                finite[i, axis] = -(ep-em)/(2*epsilon)
        np.testing.assert_allclose(finite, f, rtol=1e-6, atol=1e-13)
        np.testing.assert_allclose(f.sum(axis=0), 0, atol=1e-18)
        np.testing.assert_allclose(np.cross(q, f).sum(axis=0), 0, atol=1e-21)

    def test_sphere_energy_has_laplace_pressure_derivative(self):
        # Exact overlap volume of equal spheres separated by r <= 2R.
        h, sigma = .0008, Water().surface_tension_n_m
        def energy(radius):
            return (sigma/(5*h/48) * quad(
                lambda r: 4*math.pi*r*r*kernel(r, h) *
                (math.pi*radius*radius*r - math.pi*r**3/12), 0, h)[0])
        for radius in (.002, .004, .008):
            eps = radius * 1e-5
            pressure = ((energy(radius+eps)-energy(radius-eps))/(2*eps)
                        /(4*math.pi*radius*radius))
            self.assertAlmostEqual(pressure/(2*sigma/radius), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()

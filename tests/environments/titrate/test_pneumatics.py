"""Thermodynamic identities and bounded vent integration, not liquid evidence."""

import math
import unittest
from titrate.newton15.pneumatics import Air, IsothermalChamber, orifice_mass_flow


class PneumaticsTests(unittest.TestCase):
    def test_sealed_cycle_and_work(self):
        gas = IsothermalChamber(10.e-6)
        mass = gas.mass_kg
        expansion = gas.change_volume(20.e-6)
        self.assertAlmostEqual(gas.pressure_pa, 101325./2.)
        self.assertEqual(gas.mass_kg, mass)
        self.assertAlmostEqual(expansion['gas_work_j'], 101325.*10.e-6*math.log(2.))
        self.assertLess(gas.piston_force_n(1.e-4), 0.)
        compression = gas.change_volume(10.e-6)
        self.assertAlmostEqual(gas.pressure_pa, 101325.)
        self.assertAlmostEqual(expansion['gas_work_j']+compression['gas_work_j'], 0.)
        self.assertAlmostEqual(expansion['gauge_work_j']+compression['gauge_work_j'], 0.)

    def test_vent_direction_conservation_and_no_pressure_overshoot(self):
        for start in (0., 10000., 100000., 110000., 300000.):
            for dt in (1.e-6, .001, 1., 100.):
                gas = IsothermalChamber(10.e-6, pressure_pa=start)
                mass = gas.mass_kg
                result = gas.vent(dt, area_m2=math.pi*.00025**2, discharge_coefficient=.65)
                self.assertGreaterEqual(gas.pressure_pa, min(start, gas.air.ambient_pressure_pa)-1.e-8)
                self.assertLessEqual(gas.pressure_pa, max(start, gas.air.ambient_pressure_pa)+1.e-8)
                self.assertAlmostEqual(gas.mass_kg-mass, result['mass_exchange_kg'])
                self.assertLess(abs(result['backward_euler_residual_kg']), 1.e-10)

    def test_choked_limit_matches_independent_mach_one_expression(self):
        air = Air()
        area, cd, pressure = 1.e-7, .65, 101325.
        gamma = air.heat_capacity_ratio
        expected = cd*area*pressure/math.sqrt(air.temperature_k)*math.sqrt(gamma/air.specific_gas_constant_j_kg_k)
        expected *= ((gamma+1.)/2.)**(-(gamma+1.)/(2.*(gamma-1.)))
        for downstream in (0., pressure*.1, pressure*.5):
            self.assertAlmostEqual(orifice_mass_flow(pressure, downstream, area, cd, air), expected)
        self.assertEqual(orifice_mass_flow(pressure, pressure, area, cd, air), 0.)

    def test_invalid_volume_cannot_mutate_chamber(self):
        gas = IsothermalChamber(1.e-5)
        before = gas.mass_kg, gas.volume_m3
        for invalid in (0., -1., float('nan'), True):
            with self.assertRaises(ValueError):
                gas.change_volume(invalid)
            self.assertEqual(before, (gas.mass_kg, gas.volume_m3))


if __name__ == '__main__':
    unittest.main()

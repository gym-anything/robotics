import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from titrate.newton15.lab_config import LabConfig
from titrate.newton15.liquid_readings import valve_inventory


class ValveInventory(unittest.TestCase):
    def test_immersed_pipette_inventory_is_not_counted_again_as_source_liquid(self):
        from types import SimpleNamespace as NS
        from titrate.newton15.liquid_readings import readings
        c = LabConfig()
        names = dict(zip(('flask', 'acid_bottle', 'base_bottle', 'rinse_beaker',
            'volumetric_pipette', 'stopcock'), range(6)))
        poses = np.tile([3., 3., 3., 0., 0., 0., 1.], (6, 1))
        poses[1] = [.47, .12, .9, 0., 0., 0., 1.]
        poses[4] = [.47, .12, .9042, 0., 0., 0., 1.]
        poses[5] = [*c.valve_position, 0., 0., 0., 1.]
        q = np.array([[.47, .12, .91], [.48, .12, .91]])
        liquid = NS(water=NS(model=NS(particle_count=2), volume_per_particle_m3=1.e-9,
            config=NS(spacing_m=.001)), chemistry=NS(volume_l=1.e-6),
            world=NS(state=NS(body_q=NS(numpy=lambda: poses)),
                lab=NS(config=c, apparatus=NS(bodies=names), builder=NS())))
        result = readings(liquid, positions=q, moles=np.zeros((2, 5)))
        self.assertEqual(result['inventory']['volumetric_pipette']['parcel_count'], 1)
        self.assertEqual(result['inventory']['acid_bottle']['parcel_count'], 1)
        self.assertEqual(result['uncontained_volume_ml'], 0.)

    def test_bottle_neck_inside_the_rinse_beaker_belongs_to_the_bottle(self):
        from types import SimpleNamespace as NS
        from titrate.newton15.liquid_readings import readings
        c = LabConfig()
        names = dict(zip(('flask', 'acid_bottle', 'base_bottle', 'rinse_beaker',
            'volumetric_pipette', 'stopcock'), range(6)))
        poses = np.tile([3., 3., 3., 0., 0., 0., 1.], (6, 1))
        poses[2] = poses[3] = [.47, .12, .9, 0., 0., 0., 1.]
        poses[5] = [*c.valve_position, 0., 0., 0., 1.]
        q = np.array([[.47, .12, .95], [.498, .12, .95]])
        liquid = NS(water=NS(model=NS(particle_count=2), volume_per_particle_m3=1.e-9,
            config=NS(spacing_m=.001)), chemistry=NS(volume_l=1.e-6),
            world=NS(state=NS(body_q=NS(numpy=lambda: poses)),
                lab=NS(config=c, apparatus=NS(bodies=names), builder=NS())))
        result = readings(liquid, positions=q, moles=np.zeros((2, 5)))
        self.assertEqual(result['inventory']['base_bottle']['parcel_count'], 1)
        self.assertEqual(result['inventory']['rinse_beaker']['parcel_count'], 1)
        self.assertEqual(result['uncontained_volume_ml'], 0.)

    def test_ports_and_bore_partition(self):
        config = LabConfig()
        z = np.array([-.009, -.0075, 0., .0075, .009, .020, -.060, .003])
        points = np.array(config.valve_position)+np.column_stack((np.zeros(len(z)), np.zeros(len(z)), z))
        for angle in (0., np.pi/4, np.pi/2):
            pose = (*config.valve_position, *Rotation.from_rotvec([0., angle, 0.]).as_quat())
            regions = valve_inventory(points, pose, config)
            counts = np.sum(list(regions.values()), axis=0)
            self.assertTrue(np.all(counts <= 1))
            self.assertTrue(regions['socket_ports'][1])
            self.assertTrue(regions['socket_ports'][3])
            self.assertEqual(counts[6], 0)  # beyond the physical outlet
            self.assertEqual(counts[2], 1)  # centre of the bore at every angle
            self.assertEqual(counts[7], int(angle == 0.))  # off-axis plug solid


if __name__ == '__main__':
    unittest.main()

"""Valve geometry consistency; these checks are not a dispensing pass."""

import unittest
import numpy as np
from titrate.newton15.bored_geometry import plug_sdf, socket_sdf


class AnalyticValveTests(unittest.TestCase):

    def test_closed_mating_surfaces_have_no_fluid_gap(self):
        # The plug rotates around its cylinder axis. Outside its transverse
        # hole and the socket's vertical ports, their shared cylindrical
        # interface has zero hydraulic volume for every rotation angle.
        from scipy.spatial.transform import Rotation
        theta = np.linspace(0, 2*np.pi, 129, endpoint=False)
        r = .007+np.linspace(-40e-6, 40e-6, 33)
        points = np.array([(a*np.cos(t), .003, a*np.sin(t)) for a in r for t in theta])
        socket = socket_sdf(points)
        for angle in np.linspace(0, np.pi/2, 7):
            plug = plug_sdf(Rotation.from_euler('y', -angle).apply(points))
            self.assertLessEqual(float(np.minimum(socket, plug).max()), 1e-16)


if __name__ == '__main__':
    unittest.main()

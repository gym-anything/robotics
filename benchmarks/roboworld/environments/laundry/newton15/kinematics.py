"""Reuse the manufacturer chain; workcell base poses remain explicit."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from pencap.newton15.kinematics import Kinematics as PandaChain


class Kinematics(PandaChain):
    def __init__(self, base_y):
        super().__init__()
        self.base_y = base_y

    def forward(self, q, arm, table_top=.78):
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        t = np.eye(4)
        sign = 1 if arm == "left" else -1
        t[:3, :3] = Rotation.from_euler("z", -sign*math.pi/2).as_matrix()
        t[:3, 3] = (0., sign*self.base_y, table_top)
        for i, origin in enumerate(self.chain):
            t = t@origin
            if i < 7:
                r = np.eye(4)
                r[:3, :3] = Rotation.from_euler("z", q[i]).as_matrix()
                t = t@r
        t[:3, 3] += t[:3, :3]@np.array([0., 0., .1034])
        return t

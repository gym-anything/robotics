"""Manufacturer Panda kinematics; IK computes targets, never teleports a body."""

import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pencap.newton15.kinematics import Kinematics as PandaDescription


class Kinematics(PandaDescription):
    def forward(self, q, table_height=.75):
        t = np.eye(4)
        t[:3, :3] = Rotation.from_euler("z", math.pi).as_matrix()
        t[:3, 3] = (.54, 0., table_height+.11)
        for i, origin in enumerate(self.chain):
            t = t@origin
            if i < 7:
                r = np.eye(4)
                r[:3, :3] = Rotation.from_euler("z", q[i]).as_matrix()
                t = t@r
        t[:3, 3] += t[:3, :3]@np.array([0., 0., .1034])
        return t

    def solve(self, env, position, orientation_xyzw=None):
        names = [f"panda/panda_joint{i}" for i in range(1, 8)]
        q = env.state.joint_q.numpy()
        current = np.array([q[env.joints[n]["q"]] for n in names])
        initial = self.forward(current, env.config.table_height)
        rotation = initial[:3, :3] if orientation_xyzw is None else Rotation.from_quat(orientation_xyzw).as_matrix()
        position = np.asarray(position, float)
        def residual(x):
            t = self.forward(x, env.config.table_height)
            return np.r_[t[:3, 3]-position, .15*Rotation.from_matrix(rotation.T@t[:3, :3]).as_rotvec()]
        bounds = ([env.joints[n]["lower"] for n in names], [env.joints[n]["upper"] for n in names])
        fit = least_squares(residual, np.clip(current, *bounds), bounds=bounds,
            max_nfev=160, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = residual(fit.x)
        if np.linalg.norm(error[:3]) > .001 or np.linalg.norm(error[3:])/.15 > .02:
            raise ValueError(f"IK unreachable within tolerance ({np.linalg.norm(error[:3])*1000:.2f} mm); no action executed")
        return dict(zip(names, map(float, fit.x)))

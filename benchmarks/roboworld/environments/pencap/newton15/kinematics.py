"""Manufacturer-URDF FK/IK, evaluated without mutating the simulator."""

import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class Kinematics:
    def __init__(self):
        from titrate.newton15.assets import resolve_panda
        root = ET.parse(resolve_panda())
        self.chain = []
        for name in [f"panda_joint{i}" for i in range(1, 9)] + ["panda_hand_joint"]:
            j = root.find(f"joint[@name='{name}']")
            origin = j.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            t = np.eye(4)
            t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            t[:3, 3] = xyz
            self.chain.append(t)

    def forward(self, q, arm, table_top=.75):
        t = np.eye(4)
        sign = 1 if arm == "left" else -1
        t[:3, :3] = Rotation.from_euler("z", -sign*math.pi/2).as_matrix()
        t[:3, 3] = (0., sign*.55, table_top)
        for i, origin in enumerate(self.chain):
            t = t@origin
            if i < 7:
                r = np.eye(4)
                r[:3, :3] = Rotation.from_euler("z", q[i]).as_matrix()
                t = t@r
        # Panda grasp-target frame: 103.4 mm beyond the hand origin, z toward
        # fingertips. Finger joint travel is +/- hand y, in metres.
        t[:3, 3] += t[:3, :3]@np.array([0., 0., .1034])
        return t

    def solve(self, env, arm, position, orientation_xyzw=None):
        names = [f"{arm}/panda_joint{i}" for i in range(1, 8)]
        q = env.state.joint_q.numpy()
        current = np.array([q[env.joints[n]["q"]] for n in names])
        initial = self.forward(current, arm, env.config.table_top)
        target_r = (initial[:3, :3] if orientation_xyzw is None else
                    Rotation.from_quat(orientation_xyzw).as_matrix())
        position = np.asarray(position, float)
        def residual(x):
            t = self.forward(x, arm, env.config.table_top)
            return np.r_[t[:3, 3]-position,
                .15*Rotation.from_matrix(target_r.T@t[:3, :3]).as_rotvec()]
        bounds = ([env.joints[n]["lower"] for n in names], [env.joints[n]["upper"] for n in names])
        # Encoder positions can be microscopically beyond a soft limit; this
        # projects only the optimizer's initial guess, never a user target.
        result = least_squares(residual, np.clip(current, *bounds), bounds=bounds,
            max_nfev=160, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = residual(result.x)
        if np.linalg.norm(error[:3]) > .001 or np.linalg.norm(error[3:])/.15 > .02:
            raise ValueError(f"IK target not reached: {np.linalg.norm(error[:3])*1000:.2f} mm; no action executed")
        return dict(zip(names, map(float, result.x)))

"""Per-arm URDF kinematics from the shared robot handles (robots.controllers.kinematics.SerialKinematics). IK computes
joint targets only; the arms' position servos move the bodies.

Tool centre points:
  torch arm (UR10e): the torch work point -- on the wire axis, the nominal 19 mm contact-tip-to-work distance beyond the
      contact tip -- with z along the wire toward the work and x the tool0 x axis (the neck bends about it): a
      Cartesian command to the torch arm says where the wire meets the work and which way it points.
  hand arm (FR3): fr3_hand_tcp (between the fingertips), as in the other FR3 environments.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from robots.controllers.kinematics import SerialKinematics

from . import torch as TO


def torch_tcp_in_tool0(spec=TO.BA1):
    """The torch work point frame in the flange frame: origin at the nominal-CTWD work point, z along the wire."""
    a = math.radians(spec.neck_deg)
    R = Rotation.from_euler("x", -a).as_matrix()          # flange z -> wire axis (0, sin a, cos a)
    assert np.allclose(R @ np.array([0, 0, 1.]), spec.axis)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = spec.tcp
    return T


# the torch arm's flange link and the hand arm's tool point per arm model (the UR10e and FR3 as before). The Panda's URDF
# has no hand TCP link: the rig's tcp frame is panda_hand + 103.4 mm along z (robots/backends/newton.py, as the
# laundry/pencap/lockkey kinematics), so its chain ends at panda_hand with that offset
TORCH_LINK = {"ur10e": "tool0", "fr3": "fr3_link8", "panda": "panda_link8"}
HAND_TCP = {"fr3": ("fr3_hand_tcp", (0., 0., 0.)), "panda": ("panda_hand", (0., 0., .1034))}


class Kinematics:
    def __init__(self, world):
        self.world = world
        self.arms = {}
        for name, handle in world.rig.instances.items():
            hw = world.scene.arm_hw[name]
            (x, y), yaw = hw["mount"]
            base = np.eye(4)
            base[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
            base[:3, 3] = (x, y, hw["base_z"])
            urdf = handle.metadata["urdf"]
            names = None
            if name == world.torch_arm:
                chain = SerialKinematics(urdf, TORCH_LINK.get(hw["model"], "fr3_link8"), base=base)
                chain.chain[-1] = ("tool_offset", torch_tcp_in_tool0(world.scene.gun_spec), None)
            else:
                link, offset = HAND_TCP.get(hw["model"], ("fr3_hand_tcp", (0., 0., 0.)))
                chain = SerialKinematics(urdf, link, base=base, tip_offset=offset)
            names = [f"{name}/{j}" for j in chain.names]
            self.arms[name] = (chain, names)
        self.check()

    def check(self):
        """FK of every chain against the simulator's own link poses (mount or frame mismatch fails loudly)."""
        w = self.world
        for name, (chain, names) in self.arms.items():
            if name == w.torch_arm:
                _, _, tcp = w.torch_pose()
                fk = self.forward(name)[:3, 3]
                if np.linalg.norm(fk - tcp) > 1e-3:
                    raise RuntimeError(f"{name}: IK torch TCP {fk} differs from the simulated gun's {tcp}")
            else:
                ref = np.asarray(w.rig.frame_pose(f"{name}/tcp", w.state)["position_m"])
                fk = self.forward(name)[:3, 3]
                if np.linalg.norm(fk - ref) > 1e-3:
                    raise RuntimeError(f"{name}: IK tip {fk} differs from the rig TCP {ref}")

    def joints_now(self, arm):
        chain, names = self.arms[arm]
        q = self.world.state.joint_q.numpy()
        return np.array([q[self.world.joints[n]["q"]] for n in names])

    def forward(self, arm, q=None):
        chain, _ = self.arms[arm]
        return chain.fk(self.joints_now(arm) if q is None else np.asarray(q, float))

    def solve(self, arm, position, orientation_xyzw=None, seed=None, tol_m=1e-3, tol_rad=math.radians(1.)):
        if arm not in self.arms:
            raise ValueError(f"No arm '{arm}' in this robot deployment")
        chain, names = self.arms[arm]
        seed = self.joints_now(arm) if seed is None else np.asarray(seed, float)
        seed = np.clip(seed, chain.lower + 1e-6, chain.upper - 1e-6)
        if orientation_xyzw is None:
            orientation_xyzw = Rotation.from_matrix(chain.fk(seed)[:3, :3]).as_quat()
        r = chain.solve(position, orientation_xyzw, seed, max_evaluations=400)
        if r["position_error_m"] > tol_m or r["orientation_error_rad"] > tol_rad:
            raise ValueError(f"IK unreachable ({r['position_error_m']*1e3:.2f} mm, "
                             f"{math.degrees(r['orientation_error_rad']):.2f} deg); no action executed")
        return dict(zip(names, map(float, r["joints"])))


def torch_orientation(work_deg, travel_deg, weld_R=np.eye(3), travel_dir=np.array([1., 0, 0])):
    """Torch TCP orientation (xyzw, world) for a fillet on a horizontal T-joint: the wire axis at `work_deg` from the
    base plate in the plane normal to the weld, pushed `travel_deg` along `travel_dir` (weld frame: x along the weld,
    y out of the bracket face, z up). TCP x is chosen along the weld's -y x z ... (any roll about the wire is free;
    this keeps the gun body above and behind)."""
    th, wa = math.radians(travel_deg), math.radians(work_deg)
    z = np.array([math.sin(th), -math.cos(th)*math.cos(wa), -math.cos(th)*math.sin(wa)])   # wire toward the work
    x = np.array([1., 0, 0]) - z*z[0]
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = weld_R @ np.column_stack([x, y, z])
    return Rotation.from_matrix(R).as_quat()

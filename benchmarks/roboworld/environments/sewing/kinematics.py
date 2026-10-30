"""Per-arm URDF kinematics from the shared robot handles (robots.controllers.kinematics.SerialKinematics), as welding
and filleting. IK computes joint targets only; the arms' position servos move the bodies. The tool centre point is the
catalog's hand TCP (between the fingertips: fr3_hand_tcp / panda_hand_tcp)."""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from robots.controllers.kinematics import SerialKinematics

HAND_TCP_OFFSET = .1034   # m: robots.backends.newton's TCP for a hand link without a *_hand_tcp link (Panda)
READY_SEED = [0., -.35, 0., -2.25, 0., 1.95, .785]


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
            urdf, m = handle.metadata["urdf"], handle.model_id
            tcp_link = f"{m}_hand_tcp"
            if f'name="{tcp_link}"' in (urdf if isinstance(urdf, str) else urdf.decode()):
                chain = SerialKinematics(urdf, tcp_link, base=base)
            else:
                chain = SerialKinematics(urdf, f"{m}_hand", base=base, tip_offset=(0, 0, HAND_TCP_OFFSET))
            names = [f"{name}/{j}" for j in chain.names]
            self.arms[name] = (chain, names)
        self.check()

    def check(self):
        """FK of every chain against the simulator's own TCP frame (a mount or frame mismatch fails loudly)."""
        w = self.world
        for name in self.arms:
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

    def solve(self, arm, position, orientation_xyzw=None, seed=None, tol_m=1e-3, tol_rad=math.radians(1.), restarts=0):
        if arm not in self.arms:
            raise ValueError(f"No arm '{arm}' in this robot deployment")
        chain, names = self.arms[arm]
        seed = self.joints_now(arm) if seed is None else np.asarray(seed, float)
        seed = np.clip(seed, chain.lower + 1e-6, chain.upper - 1e-6)
        if orientation_xyzw is None:
            orientation_xyzw = Rotation.from_matrix(chain.fk(seed)[:3, :3]).as_quat()
        r = chain.solve(position, orientation_xyzw, seed, max_evaluations=400)
        if (r["position_error_m"] > tol_m or r["orientation_error_rad"] > tol_rad) and restarts:
            rng = np.random.default_rng(12345)
            for s0 in [np.array(READY_SEED)] + [rng.uniform(chain.lower, chain.upper) for _ in range(restarts)]:
                s0 = np.clip(s0, chain.lower + 1e-6, chain.upper - 1e-6)
                r2 = chain.solve(position, orientation_xyzw, s0, max_evaluations=400)
                if r2["position_error_m"] + r2["orientation_error_rad"]*0.1 < \
                        r["position_error_m"] + r["orientation_error_rad"]*0.1:
                    r = r2
                if r["position_error_m"] <= tol_m and r["orientation_error_rad"] <= tol_rad:
                    break
        if r["position_error_m"] > tol_m or r["orientation_error_rad"] > tol_rad:
            raise ValueError(f"IK unreachable ({r['position_error_m']*1e3:.2f} mm, "
                             f"{math.degrees(r['orientation_error_rad']):.2f} deg); no action executed")
        return dict(zip(names, map(float, r["joints"])))


def down(yaw_deg=0.0, pitch_deg=0.0):
    """Hand pointing down (TCP z = -world z), fingers closing along world x rotated by yaw, then tilted by pitch about
    world x so the approach axis leans forward (+y, toward the machine) and the wrist stays on the operator's side
    (clearance check E0c: straight down, the wrist meets the head near the needle) (xyzw)."""
    R = Rotation.from_euler("x", math.radians(pitch_deg))*Rotation.from_euler("xz", [math.pi, math.radians(yaw_deg)])
    return R.as_quat()

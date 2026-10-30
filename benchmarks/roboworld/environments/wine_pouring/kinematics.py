"""Per-arm URDF kinematics from the shared robot handles (barcode_scanning/label_application pattern). IK computes
joint targets only; the physical position servos move the arm."""

import math
import numpy as np
from scipy.spatial.transform import Rotation
from robots.controllers.kinematics import SerialKinematics
from . import params as P

HAND_TCP_OFFSET = .1034


class ArmKinematics:
    def __init__(self, world):
        self.world = world
        self.arms = {}
        for name, handle in world.rig.instances.items():
            (x, y), yaw = world.scene.arm_hw[name]["mount"]
            base = np.eye(4)
            base[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
            base[:3, 3] = (x, y, P.ARM_BASE_Z_M)
            urdf, m = handle.metadata["urdf"], handle.model_id
            tcp = f"{m}_hand_tcp"
            if f'name="{tcp}"' in (urdf if isinstance(urdf, str) else urdf.decode()):
                chain = SerialKinematics(urdf, tcp, base=base)
            else:
                chain = SerialKinematics(urdf, f"{m}_hand", base=base, tip_offset=(0, 0, HAND_TCP_OFFSET))
            names = [f"{name}/{j}" for j in chain.names]
            self.arms[name] = (chain, names)
            q = world.state.joint_q.numpy()
            fk = chain.fk(np.array([q[world.joints[n]["q"]] for n in names]))[:3, 3]
            ref = np.asarray(world.rig.frame_pose(f"{name}/tcp", world.state)["position_m"])
            if np.linalg.norm(fk - ref) > 1e-3:
                raise RuntimeError(f"{name}: IK tip {fk} differs from the rig TCP {ref}")

    def joints_now(self, arm):
        chain, names = self.arms[arm]
        q = self.world.state.joint_q.numpy()
        return np.array([q[self.world.joints[n]["q"]] for n in names])

    def targets_now(self, arm):
        chain, names = self.arms[arm]
        return np.array([self.world.targets[self.world.joints[n]["q"]] for n in names])

    def forward(self, arm, q=None):
        chain, _ = self.arms[arm]
        return chain.fk(self.joints_now(arm) if q is None else q)

    def solve(self, arm, position, orientation_xyzw=None, seed=None, tol_m=1e-3, tol_rad=0.0175, restarts=0):
        if arm not in self.arms:
            raise ValueError(f"No arm '{arm}' in this robot deployment")
        chain, names = self.arms[arm]
        seed = self.joints_now(arm) if seed is None else np.asarray(seed, float)
        seed = np.clip(seed, chain.lower + 1e-6, chain.upper - 1e-6)
        if orientation_xyzw is None:
            orientation_xyzw = Rotation.from_matrix(chain.fk(seed)[:3, :3]).as_quat()
        r = chain.solve(position, orientation_xyzw, seed, max_evaluations=300)
        if (r["position_error_m"] > tol_m or r["orientation_error_rad"] > tol_rad) and restarts:
            rng = np.random.default_rng(12345)
            seeds = [np.array(P.READY_POSE[:7])] + [rng.uniform(chain.lower, chain.upper) for _ in range(restarts)]
            best = None
            for s0 in seeds:
                s0 = np.clip(s0, chain.lower + 1e-6, chain.upper - 1e-6)
                cand = chain.solve(position, orientation_xyzw, s0, max_evaluations=300)
                if cand["position_error_m"] <= tol_m and cand["orientation_error_rad"] <= tol_rad:
                    dq = np.abs(np.array(cand["joints"]) - seed).max()
                    if best is None or dq < best[0]:
                        best = (dq, cand)
            if best is not None:
                r = best[1]
        if r["position_error_m"] > tol_m or r["orientation_error_rad"] > tol_rad:
            raise ValueError(f"IK unreachable ({r['position_error_m']*1e3:.2f} mm, "
                             f"{math.degrees(r['orientation_error_rad']):.2f} deg); no action executed")
        return dict(zip(names, map(float, r["joints"])))

"""Per-arm URDF kinematics from the shared robot handles (label_application.kinematics pattern). IK computes
joint targets only; it never moves a body."""

import math
import numpy as np
from scipy.spatial.transform import Rotation
from robots.controllers.kinematics import SerialKinematics

READY_SEED = [0., -.35, 0., -2.25, 0., 1.95, .785]
HAND_TCP_OFFSET = .1034


class ArmKinematics:
    def __init__(self, rig):
        self.rig = rig
        self.arms = {}
        for name, handle in rig.rig.instances.items():
            (x, y), yaw = rig.arm_hw[name]["mount"]
            base = np.eye(4)
            base[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
            base[:3, 3] = (x, y, rig.mount_z)
            urdf, m = handle.metadata["urdf"], handle.model_id
            tcp_link = f"{m}_hand_tcp"
            if f'name="{tcp_link}"' in (urdf if isinstance(urdf, str) else urdf.decode()):
                chain = SerialKinematics(urdf, tcp_link, base=base)
            else:
                chain = SerialKinematics(urdf, f"{m}_hand", base=base, tip_offset=(0, 0, HAND_TCP_OFFSET))
            names = [f"{name}/{j}" for j in chain.names]
            self.arms[name] = (chain, names)
            q = rig.state.joint_q.numpy()
            fk = chain.fk(np.array([q[rig.joints[n]["q"]] for n in names]))[:3, 3]
            ref = np.asarray(rig.rig.frame_pose(f"{name}/tcp", rig.state)["position_m"])
            if np.linalg.norm(fk - ref) > 1e-3:
                raise RuntimeError(f"{name}: IK tip {fk} differs from the rig TCP {ref}")

    def joints_now(self, arm):
        chain, names = self.arms[arm]
        q = self.rig.state.joint_q.numpy()
        return np.array([q[self.rig.joints[n]["q"]] for n in names])

    def forward(self, arm, q=None):
        chain, _ = self.arms[arm]
        return chain.fk(self.joints_now(arm) if q is None else q)

    def solve(self, arm, position, orientation_xyzw=None, seed=None, tol_m=1e-3, tol_rad=.0175, restarts=0):
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
            seeds = [np.array(READY_SEED)] + [rng.uniform(chain.lower, chain.upper) for _ in range(restarts)]
            best = None
            for s0 in seeds:
                s0 = np.clip(s0, chain.lower + 1e-6, chain.upper - 1e-6)
                c = chain.solve(position, orientation_xyzw, s0, max_evaluations=300)
                if c["position_error_m"] <= tol_m and c["orientation_error_rad"] <= tol_rad:
                    d = np.abs(np.array(c["joints"]) - seed).max()
                    if best is None or d < best[0]:
                        best = (d, c)
            if best is not None:
                r = best[1]
        if r["position_error_m"] > tol_m or r["orientation_error_rad"] > tol_rad:
            raise ValueError(f"IK unreachable ({r['position_error_m']*1e3:.2f} mm, "
                             f"{math.degrees(r['orientation_error_rad']):.2f} deg); no action executed")
        return dict(zip(names, map(float, r["joints"])))


    def cartesian_path(self, arm, start_q, poses, step_m=.01, step_rad=math.radians(8.), max_jump_rad=.35):
        """Joint targets along straight-line, slerped TCP motion through `poses` [(position, quat_xyzw), ...] from
        joint vector `start_q`: every intermediate pose (at most `step_m` and `step_rad` apart) is solved locally
        from the previous solution, with no random restarts, so the arm stays on one IK branch. Raises ValueError
        (before anything is simulated) if a pose is unreachable or any joint would jump more than `max_jump_rad`
        between neighbouring poses (a branch change). Returns a list of (joint dict, pose index) per sample."""
        chain, names = self.arms[arm]
        q = np.clip(np.asarray(start_q, float), chain.lower + 1e-6, chain.upper - 1e-6)
        T = chain.fk(q)
        p0, r0 = T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3])
        out = []
        for k, (pos, quat) in enumerate(poses):
            p1, r1 = np.asarray(pos, float), Rotation.from_quat(quat)
            ang = (r0.inv()*r1).magnitude()
            n = max(1, int(math.ceil(max(np.linalg.norm(p1 - p0)/step_m, ang/step_rad))))
            from scipy.spatial.transform import Slerp
            sl = Slerp([0., 1.], Rotation.concatenate([r0, r1]))
            for i in range(1, n + 1):
                s = i/n
                r = chain.solve(p0 + s*(p1 - p0), sl([s])[0].as_quat(), q, max_evaluations=300)
                if r["position_error_m"] > 1e-3 or r["orientation_error_rad"] > .0175:
                    raise ValueError(f"path pose {k} step {i}/{n} unreachable ({r['position_error_m']*1e3:.1f} mm)")
                qn = np.asarray(r["joints"], float)
                jump = float(np.abs(qn - q).max())
                if jump > max_jump_rad:
                    raise ValueError(f"path pose {k} step {i}/{n}: joint jump {jump:.2f} rad (IK branch change)")
                q = qn
                out.append((dict(zip(names, map(float, q))), k))
            p0, r0 = p1, r1
        return out


def tool_quat(z_axis, y_axis):
    """Orientation (xyzw) of a TCP whose approach axis is z_axis and finger-closing axis is y_axis."""
    z = np.asarray(z_axis, float); z /= np.linalg.norm(z)
    y = np.asarray(y_axis, float); y = y - (y @ z)*z; y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()

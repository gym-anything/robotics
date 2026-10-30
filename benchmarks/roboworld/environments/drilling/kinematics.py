"""Per-arm URDF kinematics from the shared robot handles (barcode_scanning / wine_pouring pattern). IK computes joint
targets only; the physical position servos move the arm. The arm bases come from the task layout's mounts.

The inverse-kinematics solve is the shared one (robots.controllers.kinematics.SerialKinematics.solve: bounded least
squares on position, 0.12 x rotation vector and 1e-5 x distance from the seed, tolerances 1e-10, at most 300 function
evaluations) with two changes that only cut its wall time (a far target took up to 1.8 s on a loaded machine, inside a
policy command, uncounted by the command floor):
  - the forward kinematics is evaluated in plain numpy with the fixed transforms merged once (FastChain.frames; the
    shared one builds a scipy Rotation per joint per call);
  - the Jacobian is analytic (FastChain.residual_jacobian) instead of the solver's 2-point finite differences, which
    cost eight forward-kinematics calls per evaluation.
The problem, its bounds, its tolerances and the cap of 300 evaluations are unchanged, and the accepted solution is
checked with the shared forward kinematics exactly as before. The cap is what bounds the wall time: it is a count, so
which targets are accepted does not depend on the machine's load. Measured with and without the two changes over the
same targets: E15 ("ik")."""

import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from robots.controllers.kinematics import SerialKinematics
from . import params as P

HAND_TCP_OFFSET = .1034
MAX_EVALUATIONS = P.IK_MAX_EVALUATIONS  # the shared solver's cap as this module has always called it
ROTATION_WEIGHT = P.IK_ROTATION_WEIGHT  # robots/controllers/kinematics.py residual(): metres per radian
SEED_WEIGHT = P.IK_SEED_WEIGHT          # the same residual's pull towards the seed


def _skew(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])


class FastChain:
    """The serial chain of a SerialKinematics (same joints, origins, axes, base and tool offset), evaluated without
    scipy objects. Consecutive fixed transforms are merged once; joint i then contributes the 4 x 4 matrix
    A_i + f1 B_i + f2 C_i, with (f1, f2) = (sin, 1 - cos) of its angle for a revolute joint (Rodrigues' formula:
    A_i the fixed transform before the joint, B_i and C_i that transform times K and K^2 of the joint axis) and
    (q, 0) for a prismatic one (B_i the fixed rotation times the axis, as a translation)."""

    def __init__(self, chain):
        self.n = len(chain.names)
        self.lower, self.upper = np.asarray(chain.lower, float), np.asarray(chain.upper, float)
        self.A, self.B, self.C = (np.zeros((self.n, 4, 4)) for _ in range(3))
        self.unit, self.scale = np.zeros((self.n, 3)), np.ones(self.n)
        self.turning = np.array([k == "revolute" for k in chain.kinds], bool)
        T = np.array(chain.base, float)
        i = 0
        for _name, origin, axis in chain.chain:
            T = T @ origin
            if axis is None:
                continue
            axis = np.asarray(axis, float)
            self.A[i] = T
            if self.turning[i]:
                self.scale[i] = float(np.linalg.norm(axis))
                self.unit[i] = axis/self.scale[i]
                K = _skew(self.unit[i])
                self.B[i, :3, :3] = T[:3, :3] @ K
                self.C[i, :3, :3] = T[:3, :3] @ K @ K
            else:
                self.unit[i] = axis
                self.B[i, :3, 3] = T[:3, :3] @ axis
            T = np.eye(4)
            i += 1
        self.tail = T                   # fixed transforms after the last joint (hand, tool offset)
        self._cache = (None, None)

    def frames(self, q):
        """Tip rotation and position, and per joint its world axis (a revolute joint's times |axis|) and origin."""
        key = q.tobytes()
        if self._cache[0] == key:
            return self._cache[1]
        a = self.scale*q
        f1 = np.where(self.turning, np.sin(a), q)
        f2 = np.where(self.turning, 1. - np.cos(a), 0.)
        M = self.A + f1[:, None, None]*self.B + f2[:, None, None]*self.C
        T = M[0]
        after = np.empty((self.n, 4, 4))
        after[0] = T
        for i in range(1, self.n):
            T = T @ M[i]
            after[i] = T
        T = T @ self.tail
        # a joint's axis and origin are unchanged by its own motion: read them after it
        axes = np.einsum("nij,nj->ni", after[:, :3, :3], self.unit)*self.scale[:, None]
        out = (T[:3, :3], T[:3, 3], axes, after[:, :3, 3])
        self._cache = (key, out)
        return out

    @staticmethod
    def rotvec(E):
        """Rotation vector of a rotation matrix (scipy's as_rotvec convention: angle in [0, pi])."""
        w = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
        s = 0.5*float(np.linalg.norm(w))                     # sin(angle)
        c = 0.5*(float(E[0, 0] + E[1, 1] + E[2, 2]) - 1.)    # cos(angle)
        if s < 1e-6 and c < 0.:                              # within a microradian of a half turn: the axis is not in w
            return Rotation.from_matrix(E).as_rotvec()
        angle = math.atan2(s, c)
        return 0.5*w if s < 1e-9 else (0.5*angle/s)*w

    def residual(self, q, p_target, R_target, seed):
        R, p, _, _ = self.frames(q)
        return np.concatenate((p - p_target, ROTATION_WEIGHT*self.rotvec(R_target @ R.T), SEED_WEIGHT*(q - seed)))

    def residual_jacobian(self, q, p_target, R_target, seed):
        """d residual / d q, analytic. Position rows: axis x (tip - joint origin) for a revolute joint, the axis for a
        prismatic one. Rotation rows: with e = rotvec(R_target R^T), a turn d about the world axis z changes e by
        -Jr^-1(e) z d (inverse right Jacobian of SO(3))."""
        R, p, axes, origins = self.frames(q)
        J = np.zeros((6 + self.n, self.n))
        turning = self.turning
        J[:3, turning] = np.cross(axes[turning], p - origins[turning]).T
        J[:3, ~turning] = axes[~turning].T
        e = self.rotvec(R_target @ R.T)
        angle = float(np.linalg.norm(e))
        S = _skew(e)
        coeff = (1./12. if angle < 1e-4 else
                 1./angle**2 - (1. + math.cos(angle))/(2.*angle*math.sin(angle)))
        Jr_inv = np.eye(3) + 0.5*S + coeff*(S @ S)
        J[3:6, turning] = -ROTATION_WEIGHT*(Jr_inv @ axes[turning].T)
        J[6:, :] = SEED_WEIGHT*np.eye(self.n)
        return J


class ArmKinematics:
    def __init__(self, world):
        self.world = world
        self.arms = {}
        self.fast = {}
        mounts = world.scene.layout["mounts"]
        for name, handle in world.rig.instances.items():
            m_ = mounts[world.scene.arm_hw[name]["mount_name"]]
            base = np.eye(4)
            base[:3, :3] = Rotation.from_euler("z", m_["yaw"]).as_matrix()
            base[:3, 3] = m_["xyz"]
            urdf, m = handle.metadata["urdf"], handle.model_id
            tcp = f"{m}_hand_tcp"
            if f'name="{tcp}"' in (urdf if isinstance(urdf, str) else urdf.decode()):
                chain = SerialKinematics(urdf, tcp, base=base)
            else:
                chain = SerialKinematics(urdf, f"{m}_hand", base=base, tip_offset=(0, 0, HAND_TCP_OFFSET))
            names = [f"{name}/{j}" for j in chain.names]
            self.arms[name] = (chain, names)
            self.fast[name] = FastChain(chain)
            q_now = self.joints_now(name)
            fk = chain.fk(q_now)
            ref = np.asarray(world.rig.frame_pose(f"{name}/tcp", world.state)["position_m"])
            if np.linalg.norm(fk[:3, 3] - ref) > 1e-3:
                raise RuntimeError(f"{name}: IK tip {fk[:3, 3]} differs from the rig TCP {ref}")
            Rf, pf, _, _ = self.fast[name].frames(q_now)     # the numpy chain must be the shared chain
            if np.abs(Rf - fk[:3, :3]).max() > 1e-9 or np.abs(pf - fk[:3, 3]).max() > 1e-9:
                raise RuntimeError(f"{name}: the fast forward kinematics differs from the shared one")

    def joints_now(self, arm):
        chain, names = self.arms[arm]
        return np.array([self.world.jq[self.world.joints[n]["q"]] for n in names])

    def targets_now(self, arm):
        chain, names = self.arms[arm]
        return np.array([self.world.targets[self.world.joints[n]["q"]] for n in names])

    def forward(self, arm, q=None):
        chain, _ = self.arms[arm]
        return chain.fk(self.joints_now(arm) if q is None else q)

    def _solve_chain(self, arm, position, orientation_xyzw, seed, max_evaluations=MAX_EVALUATIONS):
        """SerialKinematics.solve's problem and result, with the numpy forward kinematics and the analytic Jacobian."""
        chain, _ = self.arms[arm]
        fast = self.fast[arm]
        p = np.asarray(position, float)
        r = np.asarray(orientation_xyzw, float)
        seed = np.asarray(seed, float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("Finite xyz target required")
        if r.shape != (4,) or not np.isfinite(r).all() or abs(np.linalg.norm(r) - 1) > 1e-5:
            raise ValueError("Unit xyzw quaternion required")
        if seed.shape != chain.lower.shape or not np.isfinite(seed).all():
            raise ValueError("Finite seed matching chain required")
        if np.any(seed < chain.lower) or np.any(seed > chain.upper):
            raise ValueError("Seed exceeds hardware joint bounds")
        target = Rotation.from_quat(r)
        R_target = target.as_matrix()
        fit = least_squares(fast.residual, seed, jac=fast.residual_jacobian, bounds=(chain.lower, chain.upper),
                            max_nfev=max_evaluations, ftol=1e-10, xtol=1e-10, gtol=1e-10, args=(p, R_target, seed))
        final = chain.fk(fit.x)                              # accepted or refused on the shared forward kinematics
        error = float(np.linalg.norm(final[:3, 3] - p))
        angular = float((target*Rotation.from_matrix(final[:3, :3]).inv()).magnitude())
        return dict(joints=fit.x.tolist(), position_error_m=error, orientation_error_rad=angular,
                    converged=bool(error < 0.001 and angular < 0.01745329252), evaluations=int(fit.nfev),
                    note="Bounded kinematic target only; not a collision-free path or physical move")

    def solve(self, arm, position, orientation_xyzw=None, seed=None, tol_m=1e-3, tol_rad=0.0175, restarts=0,
              reference=False):
        """Joint targets for a TCP pose, nearest the seed (default: the arm's present joints); ValueError when the
        pose is not reached within tol_m and tol_rad. reference=True runs the shared solver unchanged (2-point
        numerical Jacobian): the comparison the runtime experiment records."""
        if arm not in self.arms:
            raise ValueError(f"No arm '{arm}' in this robot deployment")
        chain, names = self.arms[arm]
        seed = self.joints_now(arm) if seed is None else np.asarray(seed, float)
        seed = np.clip(seed, chain.lower + 1e-6, chain.upper - 1e-6)
        if orientation_xyzw is None:
            orientation_xyzw = Rotation.from_matrix(chain.fk(seed)[:3, :3]).as_quat()

        def one(s0):
            if reference:
                return chain.solve(position, orientation_xyzw, s0, max_evaluations=MAX_EVALUATIONS)
            return self._solve_chain(arm, position, orientation_xyzw, s0)
        r = one(seed)
        if (r["position_error_m"] > tol_m or r["orientation_error_rad"] > tol_rad) and restarts:
            rng = np.random.default_rng(12345)
            seeds = [np.array(P.READY_POSE[:7])] + [rng.uniform(chain.lower, chain.upper) for _ in range(restarts)]
            best = None
            for s0 in seeds:
                s0 = np.clip(s0, chain.lower + 1e-6, chain.upper - 1e-6)
                cand = one(s0)
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

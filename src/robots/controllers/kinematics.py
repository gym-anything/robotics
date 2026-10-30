"""Robot-asset FK and bounded IK. Produces actuator targets, not task behavior."""

import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    t[:3, 3] = xyz
    return t


class SerialKinematics:
    def __init__(self, payload, tip, *, base, tip_offset=(0, 0, 0)):
        xml = ET.fromstring(payload)
        by_child = {j.find("child").get("link"): j for j in xml.findall("joint")}
        node = tip
        chain = []
        while node in by_child:
            joint = by_child[node]
            chain.append(joint)
            node = joint.find("parent").get("link")
        self.chain = []
        self.names = []
        self.lower = []
        self.upper = []
        self.kinds = []
        for joint in reversed(chain):
            kind = joint.get("type")
            origin = joint.find("origin")
            xyz = (
                np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            rpy = (
                np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            if kind not in ("fixed", "revolute", "prismatic"):
                raise ValueError(f"Unsupported serial chain joint: {kind}")
            axis = None
            if kind != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                limit = joint.find("limit")
                self.lower.append(float(limit.get("lower")))
                self.upper.append(float(limit.get("upper")))
                self.names.append(joint.get("name"))
                self.kinds.append(kind)
            self.chain.append((joint.get("name"), transform(xyz, rpy), axis))
        self.chain.append(("tool_offset", transform(tip_offset), None))
        self.lower = np.asarray(self.lower)
        self.upper = np.asarray(self.upper)
        self.base = np.asarray(base)

    def fk(self, q, *, through=None):
        q = np.asarray(q, float)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise ValueError("Finite joint vector matching the selected chain required")
        t = self.base.copy()
        i = 0
        for name, origin, axis in self.chain:
            t = t @ origin
            if axis is not None:
                r = np.eye(4)
                if self.kinds[i] == "revolute":
                    r[:3, :3] = Rotation.from_rotvec(axis * q[i]).as_matrix()
                else:
                    r[:3, 3] = axis * q[i]
                t = t @ r
                i += 1
            if name == through:
                break
        return t

    def solve(self, position, orientation_xyzw, seed, *, max_evaluations=160):
        p = np.asarray(position, float)
        r = np.asarray(orientation_xyzw, float)
        seed = np.asarray(seed, float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("Finite xyz target required")
        if (
            r.shape != (4,)
            or not np.isfinite(r).all()
            or abs(np.linalg.norm(r) - 1) > 1e-5
        ):
            raise ValueError("Unit xyzw quaternion required")
        if seed.shape != self.lower.shape or not np.isfinite(seed).all():
            raise ValueError("Finite seed matching chain required")
        if np.any(seed < self.lower) or np.any(seed > self.upper):
            raise ValueError("Seed exceeds hardware joint bounds")
        target = Rotation.from_quat(r)

        def residual(q):
            t = self.fk(q)
            angle = (target * Rotation.from_matrix(t[:3, :3]).inv()).as_rotvec()
            return np.r_[t[:3, 3] - p, 0.12 * angle, 1e-5 * (q - seed)]

        fit = least_squares(
            residual,
            seed,
            bounds=(self.lower, self.upper),
            max_nfev=max_evaluations,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        final = self.fk(fit.x)
        error = float(np.linalg.norm(final[:3, 3] - p))
        angular = float(
            (target * Rotation.from_matrix(final[:3, :3]).inv()).magnitude()
        )
        return dict(
            joints=fit.x.tolist(),
            position_error_m=error,
            orientation_error_rad=angular,
            converged=bool(error < 0.001 and angular < 0.01745329252),
            evaluations=int(fit.nfev),
            note="Bounded kinematic target only; not a collision-free path or physical move",
        )

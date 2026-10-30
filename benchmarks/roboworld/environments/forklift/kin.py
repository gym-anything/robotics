"""Kinematics of the seated operator on a private MjData copy of the world's MuJoCo model (reads the current state,
never writes it). Used to build start postures and by the ASSISTED scripted motions (demos and probes), never by the
policy interface. Collision clearance is checked with MuJoCo's own geometry (mj_geomDistance) on the scratch data.
"""

import numpy as np


class Kin:
    def __init__(self, world):
        import mujoco
        self.w, self.mj = world, mujoco
        self.model = world.solver.mj_model
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.jname = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "": j for j in range(m.njnt)}
        self.bname = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "": i for i in range(m.nbody)}
        op = world.operator
        self.qadr = {}
        self.limits = {}
        for n, a in op.joints.items():
            short = n.split("/")[-1]
            hit = [j for k, j in self.jname.items() if k.endswith(short)]
            if len(hit) != 1:
                raise RuntimeError(f"joint {short}: {hit}")
            self.qadr[n] = int(m.jnt_qposadr[hit[0]])
            self.limits[n] = (a["lower"], a["upper"])
        pel = world.mj_body(op.pelvis)
        self.free_adr = int(m.jnt_qposadr[int(m.body_jntadr[pel])])

    def body(self, short):
        hit = [i for k, i in self.bname.items() if k.endswith(short)]
        if len(hit) != 1:
            raise RuntimeError(f"body {short}: {[k for k in self.bname if k.endswith(short)]}")
        return hit[0]

    def sync(self, joints=None, keep_base=False):
        """Load the world's current state, then override joints; keep_base keeps this data's floating-base pose."""
        w = self.w
        base = self.data.qpos[self.free_adr:self.free_adr + 7].copy() if keep_base else None
        w.solver._update_mjc_data(self.data, w.model, w.state)
        if base is not None:
            self.data.qpos[self.free_adr:self.free_adr + 7] = base
        for n, v in (joints or {}).items():
            self.data.qpos[self.qadr[n]] = v
        self.mj.mj_kinematics(self.model, self.data)

    def set_base(self, p, R):
        """Put this scratch data's floating base at a world pose (planning with the trunk held at a prescribed pose:
        assist.TrunkSupport); the world's state is not touched."""
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(np.asarray(R, float)).as_quat()
        self.data.qpos[self.free_adr:self.free_adr + 3] = np.asarray(p, float)
        self.data.qpos[self.free_adr + 3:self.free_adr + 7] = (q[3], q[0], q[1], q[2])
        self.mj.mj_kinematics(self.model, self.data)

    def pose(self, b):
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def point(self, b, local):
        p, R = self.pose(b)
        return p + R @ np.asarray(local, float)

    def joints_now(self, names):
        return {n: float(self.data.qpos[self.qadr[n]]) for n in names}

    def distance(self, g, h, cap=0.05):
        """Signed distance between two geoms, capped at `cap`. mj_geomDistance returns exactly 0.0 for some separated
        mesh pairs whenever distmax exceeds their true distance (E14b: torso / seat backrest read 0.0 at distmax 0.06
        and 0.02 at distmax 0.02; independent of ccd_iterations and ccd_tolerance), which a clearance term reads as
        touching. For such a pair the distance is bracketed by bisection on distmax: below the true distance the
        function returns distmax itself."""
        f = self.mj.mj_geomDistance
        d = f(self.model, self.data, g, h, cap, None)
        if d != 0.0:
            return d
        lo, hi = 0.0, cap
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            r = f(self.model, self.data, g, h, mid, None)
            if r == mid:
                lo = mid
            elif r == 0.0:
                hi = mid
            else:
                return r
        return lo

    def solve(self, names, targets, nominal=None, reg=0.01, pairs=(), clearance=0.008, x0=None, max_nfev=200, robust=False):
        """Least-squares IK over `names`. targets: list of (body, local point, world point, weight) or
        (body, 'axis', local axis, world axis, weight). Regularised toward `nominal` (default: current). Returns
        (joint dict, residual dict). robust: clearance distances through `distance` (the default keeps the plain
        mj_geomDistance the start posture was solved with)."""
        from scipy.optimize import least_squares
        adr = np.array([self.qadr[n] for n in names])
        lo = np.array([self.limits[n][0] for n in names]) + 1e-3
        hi = np.array([self.limits[n][1] for n in names]) - 1e-3
        base = self.data.qpos.copy()
        q0 = np.clip(base[adr] if x0 is None else np.asarray(x0, float), lo, hi)
        qn = np.array([(nominal or {}).get(n, base[a]) for n, a in zip(names, adr)])

        def res(q):
            self.data.qpos[:] = base
            self.data.qpos[adr] = q
            self.mj.mj_kinematics(self.model, self.data)
            r = []
            for t in targets:
                if isinstance(t[1], str):
                    b, _, la, wa, wgt = t
                    p, R = self.pose(b)
                    r.append(wgt * (R @ np.asarray(la, float) - np.asarray(wa, float)))
                elif isinstance(t[2], tuple) and len(t[2]) == 3 and t[2][0] == "plane":   # point on a plane
                    b, lp, (_, p0, nrm), wgt = t
                    r.append(np.atleast_1d(wgt * float(np.asarray(nrm) @ (self.point(b, lp) - np.asarray(p0)))))
                elif len(t) == 5:                    # per-axis weights in the frame R (columns = axes)
                    b, lp, wp_, wgt, Rf = t
                    r.append(np.asarray(wgt) * (np.asarray(Rf).T @ (self.point(b, lp) - np.asarray(wp_, float))))
                else:
                    b, lp, wp_, wgt = t
                    r.append(wgt * (self.point(b, lp) - np.asarray(wp_, float)))
            r.append(reg * (q - qn))
            if pairs:
                if robust:
                    d = np.array([self.distance(g, h, 0.05) for g, h in pairs])
                else:
                    d = np.array([self.mj.mj_geomDistance(self.model, self.data, g, h, 0.05, None) for g, h in pairs])
                r.append(3.0 * np.maximum(0.0, clearance - d))
            return np.concatenate(r)
        sol = least_squares(res, q0, bounds=(lo, hi), max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
        q = sol.x
        self.data.qpos[:] = base
        self.data.qpos[adr] = q
        self.mj.mj_kinematics(self.model, self.data)
        errs = []
        for t in targets:
            if isinstance(t[1], str):
                b, _, la, wa, wgt = t
                errs.append(float(np.degrees(np.arccos(np.clip((self.pose(b)[1] @ np.asarray(la)) @ np.asarray(wa), -1, 1)))))
            elif isinstance(t[2], tuple) and len(t[2]) == 3 and t[2][0] == "plane":
                b, lp, (_, p0, nrm), wgt = t
                errs.append(float(abs(np.asarray(nrm) @ (self.point(b, lp) - np.asarray(p0)))))
            else:
                b, lp, wp_ = t[0], t[1], t[2]
                e = self.point(b, lp) - np.asarray(wp_)
                if len(t) == 5:
                    e = (np.asarray(t[3]) > 0) * (np.asarray(t[4]).T @ e)
                errs.append(float(np.linalg.norm(e)))
        return {n: float(v) for n, v in zip(names, q)}, errs

    def min_distance(self, geoms_a, geoms_b, cap=0.3):
        d = [self.mj.mj_geomDistance(self.model, self.data, g, h, cap, None) for g in geoms_a for h in geoms_b]
        return min(d) if d else cap

    def geom_pairs(self, bodies_a, bodies_b):
        m = self.model
        A = set(bodies_a); B = set(bodies_b)
        return [(g, h) for g in range(m.ngeom) if int(m.geom_bodyid[g]) in A
                for h in range(m.ngeom) if int(m.geom_bodyid[h]) in B]

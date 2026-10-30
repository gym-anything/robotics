"""ASSISTED fixtures for probes and demonstrations (never reachable from the policy API).

PhantomMount: a six-axis servo on one tool body -- a spring-damper pulling a body-fixed point toward a scripted world
trajectory and the body's orientation toward a scripted rotation, with the body's own weight (incl. lumped paint)
compensated, and optionally a pure force along one world axis instead of the spring there (force-controlled
pressing). It is a 'phantom hand': labelled ASSISTED in every caption, recorded per tick.
PhantomPress: a pure force on a body point (e.g. a fingertip-free press on the aerosol button).
Fixtures add to world.body_wrench (the paint applicators' contact wrenches are already in it).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from painting import params as P


def chain(*fixtures):
    def f(world):
        for fx in fixtures:
            fx.apply(world)
        return world.body_wrench
    return f


class PhantomMount:
    def __init__(self, world, bodies, point_local, target_fn, *, k=2500.0, zeta=1.0, k_rot=25.0, force_axis=None,
                 force_fn=None, label="phantom mount", f_max=60.0):
        """bodies: the mounted body first, then bodies carried with it (their weight is compensated too; the roller's
        cover). target_fn(t) -> (position (3,), Rotation or None)."""
        self.w = world
        self.bodies = list(bodies)
        self.body = self.bodies[0]
        self.point = np.asarray(point_local, float)
        self.target_fn, self.force_fn = target_fn, force_fn
        self.k, self.zeta, self.k_rot = k, zeta, k_rot
        self.axis = None if force_axis is None else np.asarray(force_axis, float)/np.linalg.norm(force_axis)
        self.label = label
        self.f_max = f_max
        self.log = []

    def _mass(self):
        m = self.w.mjm
        return sum(float(m.body_mass[self.w._mj_body[b]]) for b in self.bodies)

    def apply(self, world):
        import mujoco
        w = world
        t = w.sim_time
        m = self._mass()
        k = w._mj_body[self.body]
        p, R = w.body_pose(self.body)
        at = p + R @ self.point
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(w.mjm, w.mjd, mujoco.mjtObj.mjOBJ_BODY, k, v6, 0)
        om, v_o = v6[:3], v6[3:]
        v_at = v_o + np.cross(om, at - np.array(w.mjd.xpos[k]))
        target, rot = self.target_fn(t)
        c = 2*self.zeta*np.sqrt(self.k*m)
        spring = self.k*(np.asarray(target) - at)
        sn = np.linalg.norm(spring)
        if sn > self.f_max:                    # a hand does not yank: the servo's spring force is capped
            spring *= self.f_max/sn
        F = spring - c*v_at
        if self.axis is not None:
            F -= (F @ self.axis)*self.axis
            F += (self.force_fn(t) if self.force_fn else 0.0)*self.axis
        com = np.array(w.mjd.xipos[k])
        tau = np.cross(at - com, F)
        # weight compensation acts at each body's own centre of mass (the mounted body's and the carried ones'); applied
        # at the mount point, it would put a spurious moment (mount point - COM) x m g on the tool that the rotational
        # servo resists only by tilting (a held roller's axle 3.5 deg, P3)
        F = F + np.array([0, 0, m*P.GRAVITY])
        for b in self.bodies[1:]:
            kb = w._mj_body[b]
            tau += np.cross(np.array(w.mjd.xipos[kb]) - com, np.array([0, 0, float(w.mjm.body_mass[kb])*P.GRAVITY]))
        if rot is not None:
            # inertia-scaled rotational servo, tau = I_world (wn^2 err - 2 zeta wn omega): the same natural frequency about
            # every axis (a brush's long axis has ~1/40 of its transverse inertia; an isotropic damper went unstable)
            Rt = rot.as_matrix() if hasattr(rot, "as_matrix") else np.asarray(rot)
            err = Rotation.from_matrix(Rt @ R.T).as_rotvec()
            Xi = np.asarray(w.mjd.ximat[k]).reshape(3, 3)
            Iw = Xi @ np.diag(np.asarray(w.mjm.body_inertia[k], float)) @ Xi.T
            for b in self.bodies[1:]:
                kb = w._mj_body[b]
                Xb = np.asarray(w.mjd.ximat[kb]).reshape(3, 3)
                Iw = Iw + Xb @ np.diag(np.asarray(w.mjm.body_inertia[kb], float)) @ Xb.T
            wn = self.k_rot
            tau += Iw @ (wn*wn*err - 2*self.zeta*wn*om)
        w.body_wrench[self.body, :3] += F
        w.body_wrench[self.body, 3:] += tau
        if w.tick % 20 == 0:
            self.log.append(dict(t=round(t, 4), force=np.round(F, 3).tolist(), target=np.round(target, 4).tolist(),
                                 point=np.round(at, 4).tolist()))
        return w.body_wrench


class PhantomPress:
    def __init__(self, world, body, point_local, force_fn, label="phantom press"):
        self.w, self.body, self.point, self.force_fn, self.label = world, body, np.asarray(point_local, float), force_fn, label
        self.log = []

    def apply(self, world):
        w = world
        p, R = w.body_pose(self.body)
        F = np.asarray(self.force_fn(w.sim_time), float)
        k = w._mj_body[self.body]
        at = p + R @ self.point
        w.body_wrench[self.body, :3] += F
        w.body_wrench[self.body, 3:] += np.cross(at - np.array(w.mjd.xipos[k]), F)
        if w.tick % 20 == 0:
            self.log.append(dict(t=round(w.sim_time, 4), force=np.round(F, 3).tolist()))
        return w.body_wrench


def smooth(a, b, s):
    s = min(1.0, max(0.0, s))
    s = s*s*(3 - 2*s)
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float))*s

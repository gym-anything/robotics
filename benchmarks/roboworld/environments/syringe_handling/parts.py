"""Building blocks for the handled objects: poses, links, joints bound to MuJoCo, primitive shapes.

Helpers as in ingredient_mixing/tools.py and drilling/tools.py (the same Newton builder calls), kept in one module.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation


def wq(R):
    q = R.as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def rot_z_to(axis):
    """Rotation taking +z to `axis`."""
    a = dict(x=(1., 0., 0.), y=(0., 1., 0.), z=(0., 0., 1.)).get(axis, axis) if isinstance(axis, str) else axis
    a = np.asarray(a, float)/np.linalg.norm(a)
    z = np.array([0., 0., 1.])
    c = float(z @ a)
    if c > 1 - 1e-12:
        return Rotation.identity()
    if c < -1 + 1e-12:
        return Rotation.from_euler("x", math.pi)
    v = np.cross(z, a)
    return Rotation.from_rotvec(v/np.linalg.norm(v)*math.acos(c))


class JointRef:
    """A Newton joint with its coordinate, dof, MuJoCo joint and actuator indices, bound after finalize."""

    def __init__(self, index, name):
        self.index, self.name = int(index), name
        self.q = self.dof = self.mj = self.act = None

    def bind(self, world):
        m = world.model
        self.q = int(m.joint_q_start.numpy()[self.index])
        self.dof = int(m.joint_qd_start.numpy()[self.index])
        self.mj = world.mj_joint_of_dof[self.dof]
        mjm = world.solver.mj_model
        acts = [a for a in range(mjm.nu) if int(mjm.actuator_trnid[a][0]) == self.mj]
        self.act = acts[0] if acts else None
        return self

    def pos(self, w):
        return float(w.jq[self.q])

    def vel(self, w):
        return float(w.jqd[self.dof])


def link(sc, name, pose, mass=None, com=None, inertia=None):
    p, R = pose
    kw = {}
    if mass is not None:
        kw.update(mass=float(mass))
    if com is not None:
        kw.update(com=sc.wp.vec3(*map(float, com)))
    if inertia is not None:
        I = np.asarray(inertia, float)
        I = np.diag(I) if I.ndim == 1 else I
        kw.update(inertia=sc.wp.mat33(*I.ravel().tolist()), lock_inertia=True)
    return sc.b.add_link(xform=sc.wp.transform(tuple(map(float, p)), wq(R)), label=name, **kw)


def rel_xform(sc, parent_pose, child_pose):
    pp, Rp = parent_pose
    pc, Rc = child_pose
    R = Rp.inv()*Rc
    return sc.wp.transform(tuple(map(float, Rp.inv().apply(np.asarray(pc, float) - np.asarray(pp, float)))), wq(R))


def compose(pose, offset, rot=None):
    p, R = pose
    return (np.asarray(p, float) + R.apply(offset), R*(rot if rot is not None else Rotation.identity()))


def capsule(sc, body, name, p0, p1, radius, material, collide=True, **kw):
    """A capsule between two points of the body frame."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    R = rot_z_to(p1 - p0)
    cfg = sc.cfg(**kw) if collide else sc.visual_cfg()
    i = sc.b.add_shape_capsule(body, xform=sc.wp.transform(tuple(0.5*(p0 + p1)), wq(R)), radius=float(radius),
                               half_height=float(0.5*np.linalg.norm(p1 - p0)), cfg=cfg, label=name)
    sc.materials[i] = material
    if not collide:
        sc.visual.add(i)
    return i


def cylinder(sc, body, name, pos, radius, half_h, axis, material, collide=True, **kw):
    cfg = sc.cfg(**kw) if collide else sc.visual_cfg()
    i = sc.b.add_shape_cylinder(body, xform=sc.wp.transform(tuple(map(float, pos)), wq(rot_z_to(axis))),
                                radius=float(radius), half_height=float(half_h), cfg=cfg, label=name)
    sc.materials[i] = material
    if not collide:
        sc.visual.add(i)
    return i


def box(sc, body, name, pos, half, material, rot=None, collide=True, **kw):
    return sc.box(name, pos, half, material, body=body, rot=wq(rot) if isinstance(rot, Rotation) else rot, collide=collide, **kw)


def sphere(sc, body, name, pos, radius, material, collide=True, **kw):
    cfg = sc.cfg(**kw) if collide else sc.visual_cfg()
    i = sc.b.add_shape_sphere(body, xform=sc.wp.transform(tuple(map(float, pos)), sc.wp.quat_identity()),
                              radius=float(radius), cfg=cfg, label=name)
    sc.materials[i] = material
    if not collide:
        sc.visual.add(i)
    return i


def world_pose(w, body):
    """(position, Rotation) of a Newton body from the world's last refresh."""
    q = w.bq[int(body)]
    return q[:3].copy(), Rotation.from_quat(q[3:])


def to_world(pose, p_local):
    return pose[0] + pose[1].apply(p_local)


def quat_wxyz(R):
    q = R.as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

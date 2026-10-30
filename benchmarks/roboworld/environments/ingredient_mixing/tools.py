"""Vessels, hand tools and appliances: rigid bodies, the solids and wires the mixture sees, and the mechanisms.

Every object is described once here. For the rigid solver it is convex pieces (hulls, boxes, capsules); for the
continuum it is signed-distance colliders (continuum.add_collider) and, where a part is thinner than a grid cell (whisk
wires, beater strips), chains of wire markers that exchange momentum with the material by the cylinder drag laws.

Tool frame: origin at the handle's butt, +z from the butt toward the working end. A tool standing head-down in the
utensil crock has its +z along world -z.

  Vessel      a body of revolution from its inner profile (geometry.py): bowl, saucepan, pitcher, crock
  whisk       handle + sprung cage (two bending axes at the ferrule) + wire markers
  spatula     handle with the blade's root + sprung blade tip
  spoon, stirrer   rigid
  Mixer       housing, loop handle, slide switch with OFF and five detents, two counter-rotating beaters on a motor with
              a straight torque-speed line
  HobKnob     a detent for off and one for each of the hob's twenty levels; the level sets its power (heat.py)
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import geometry as G
from . import continuum as CT

RPM = 2*math.pi/60


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


def hull(sc, body, name, points, material, hidden=True, **kw):
    import trimesh
    h = trimesh.convex.convex_hull(np.asarray(points, float))
    mesh = sc.newton.Mesh(np.asarray(h.vertices, np.float32), np.asarray(h.faces, np.int32).ravel())
    i = sc.b.add_shape_convex_hull(body, mesh=mesh, xform=sc.wp.transform_identity(),
                                   cfg=sc.cfg(gap=P.MESH_CONTACT_GAP_M, **kw), label=name)
    sc.materials[i] = "collision_hidden" if hidden else material
    if hidden:
        sc.visual.add(i)
    return i


def sphere(sc, body, name, pos, radius, material, collide=True, **kw):
    cfg = sc.cfg(**kw) if collide else sc.visual_cfg()
    i = sc.b.add_shape_sphere(body, xform=sc.wp.transform(tuple(map(float, pos)), sc.wp.quat_identity()),
                              radius=float(radius), cfg=cfg, label=name)
    sc.materials[i] = material
    if not collide:
        sc.visual.add(i)
    return i


def disc(sc, body, name, centre, radius, half_h, material, hidden=True, **kw):
    """A round slab about the body's z as six crossed boxes: a regular dodecagon whose flats lie on the circle.
    (Contacts stay among boxes, capsules and spheres, MuJoCo's analytic pairs: a spoon standing on a convex-hull floor
    was kicked to 3.3 m/s by a contact that appeared 0.75 mm deep.)"""
    ids = []
    for k in range(6):
        R = Rotation.from_euler("z", k*math.pi/6)
        i = sc.box(f"{name}_{k}", centre, (radius, radius*math.tan(math.pi/12), half_h), material, body=body, rot=wq(R), **kw)
        if hidden:
            sc.materials[i] = "collision_hidden"
            sc.visual.add(i)
        ids.append(i)
    return ids


# ====================================================================== vessels
def shell_mass_properties(profile, mass, z_shift=0.0):
    """Mass, centre of mass and principal inertia (about the centre of mass) of a thin shell of revolution."""
    p = np.asarray(profile, float)
    seg = np.diff(p, axis=0)
    ds = np.linalg.norm(seg, axis=1)
    mid = 0.5*(p[:-1] + p[1:])
    dA = 2*math.pi*mid[:, 0]*ds
    dm = mass*dA/dA.sum()
    zc = float((dm*mid[:, 1]).sum()/mass)
    izz = float((dm*mid[:, 0]**2).sum())
    ixx = float((dm*(0.5*mid[:, 0]**2 + (mid[:, 1] - zc)**2)).sum())
    return mass, (0., 0., zc + z_shift), (ixx, ixx, izz)


class Vessel:
    """A body of revolution open at the top. Body frame: origin at the centre of the inside bottom, z up."""

    def __init__(self, sc, name, profile, *, wall_rigid, mass, pos, material, mu, yaw=0.0, sectors=24, bands=5,
                 bead=0.0, static=False, handle=None, divider=0, render_wall=None):
        self.name, self.profile, self.kind = name, np.asarray(profile, float), material
        self.wall_rigid, self.mass = wall_rigid, mass
        self.rim_r, self.depth = float(self.profile[-1, 0]), float(self.profile[-1, 1])
        self.render_wall = render_wall if render_wall is not None else wall_rigid
        self.handle = handle
        pose = (np.asarray(pos, float), Rotation.from_euler("z", yaw))
        m, com, inertia = shell_mass_properties(self.profile, mass)
        if handle:
            hm = handle["mass"]
            hc = np.asarray(handle["p0"], float) + 0.5*(np.asarray(handle["p1"], float) - np.asarray(handle["p0"], float))
            c_all = (m*np.asarray(com) + hm*hc)/(m + hm)
            L = float(np.linalg.norm(np.asarray(handle["p1"]) - np.asarray(handle["p0"])))
            I = np.diag(inertia) + m*(np.dot(com - c_all, com - c_all)*np.eye(3) - np.outer(com - c_all, com - c_all)) \
                + hm*(np.dot(hc - c_all, hc - c_all)*np.eye(3) - np.outer(hc - c_all, hc - c_all)) + np.eye(3)*hm*L*L/12
            m, com, inertia = m + hm, tuple(c_all), I
        self.total_mass, self.com = m, np.asarray(com, float)
        if static:
            self.body = link(sc, name, pose, mass=m, com=com, inertia=inertia)
            self.joint = sc.b.add_joint_fixed(-1, self.body, parent_xform=sc.wp.transform(tuple(pose[0]), wq(pose[1])),
                                              child_xform=sc.wp.transform_identity(), label=f"{name}_fixed")
        else:
            self.body = link(sc, name, pose, mass=m, com=com, inertia=inertia)
            self.joint = sc.b.add_joint_free(self.body, label=f"{name}_free")
        sc.b.add_articulation([self.joint], label=f"{name}_assembly")
        sc.bodies[name] = self.body
        # rigid pieces, all boxes: the base disc, then the wall as flat facets (bands along the profile x sectors
        # around), each a slab whose inner face is tangent to the true inner surface at its middle
        prof = self.profile[1:]
        sl = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
        cuts = np.linspace(0, sl[-1], bands + 1)
        pts = np.column_stack([np.interp(cuts, sl, prof[:, 0]), np.interp(cuts, sl, prof[:, 1])])
        if bead > 0:
            pts = np.vstack([pts[:-1], [np.interp(self.depth - 0.006, prof[:, 1], prof[:, 0]), self.depth - 0.006], pts[-1]])
        disc(sc, self.body, f"{name}_base", (0., 0., -0.5*wall_rigid), pts[0, 0] + (wall_rigid if abs(pts[1, 0] - pts[0, 0])
             < 1e-6 else 0.), 0.5*wall_rigid, material, density=0., mu=mu)
        self.n_boxes = 6
        for k in range(len(pts) - 1):
            p0, p1 = pts[k], pts[k + 1]
            seg = float(np.linalg.norm(p1 - p0))
            d = (p1 - p0)/seg
            nrm = np.array([d[1], -d[0]])
            t = bead if (bead > 0 and k == len(pts) - 2) else wall_rigid
            mid = 0.5*(p0 + p1) + 0.5*t*nrm
            half_w = (max(p0[0], p1[0]) + t)*math.tan(math.pi/sectors) + 2e-4
            for q in range(sectors):
                am = 2*math.pi*(q + 0.5)/sectors
                ca, sa = math.cos(am), math.sin(am)
                R = Rotation.from_matrix(np.column_stack([(-sa, ca, 0.), (d[0]*ca, d[0]*sa, d[1]),
                                                          (nrm[0]*ca, nrm[0]*sa, nrm[1])]))
                i = sc.box(f"{name}_w{k}", (mid[0]*ca, mid[0]*sa, mid[1]), (half_w, 0.5*seg + 2e-4, 0.5*t), material,
                           body=self.body, rot=wq(R), density=0., mu=mu)
                sc.materials[i] = "collision_hidden"
                sc.visual.add(i)
                self.n_boxes += 1
        if divider:
            for k in range(divider):                      # radial walls of a utensil crock's insert
                a = 2*math.pi*k/divider + math.pi/2
                R = Rotation.from_euler("z", a)
                r_in = self.rim_r*0.5
                sc.box(f"{name}_divider{k}", tuple(R.apply((r_in, 0., 0.5*self.depth))), (r_in, 0.0015, 0.5*self.depth),
                       material, body=self.body, rot=wq(R), density=0., mu=mu)
        if handle:
            capsule(sc, self.body, f"{name}_handle", handle["p0"], handle["p1"], handle["r"], handle.get("material", material),
                    density=0., mu=handle.get("mu", mu), gap=P.MESH_CONTACT_GAP_M)
        self.sdf = None

    def collider(self, cont, shell_cells=5.0, mu_dry=0.19):
        """Register the vessel with the continuum: the true inner surface, thickened outward."""
        shell = shell_cells*cont.dx
        self.sdf = G.revolved_sdf([G.shell_polygon(self.profile, shell)], h=cont.dx/4)
        self.col = cont.add_collider(CT.KIND_REVOLVED, sdf=self.sdf, body=self.body, mu_dry=mu_dry, name=self.name)
        return self.col

    def inside(self, local, margin=0.0):
        """Whether points (n, 3) in the vessel frame are inside its interior (below the rim, within the inner surface)."""
        r = np.hypot(local[:, 0], local[:, 1])
        z = local[:, 2]
        rr = np.interp(z, self.profile[1:, 1], self.profile[1:, 0], left=self.profile[1, 0], right=self.profile[-1, 0])
        return (z > -margin) & (z < self.depth + margin) & (r < rr + margin)


# ====================================================================== hand tools
class HandTool:
    kind = "tool"

    def __init__(self):
        self.bodies, self.j = [], {}
        self.colliders, self.markers = [], []      # continuum: dict(kind, body, pos, quat, size) / dict(body, x, t, len, dia)

    def bind(self, world):
        self.w = world
        for j in self.j.values():
            j.bind(world)

    def pre_step(self, w):
        pass

    def post_step(self, w):
        pass

    def state(self):
        return {}

    def snapshot(self):
        return {k: v for k, v in self.__dict__.items() if k.startswith("s_")}

    def restore(self, snap):
        self.__dict__.update(snap)

    def register(self, cont, mu_dry=0.19):
        self.col_ids = []
        for c in self.colliders:
            self.col_ids.append(cont.add_collider(c["kind"], body=c["body"], pos=c["pos"], quat=c.get("quat", (0, 0, 0, 1)),
                                                  size=c["size"], mu_dry=c.get("mu_dry", mu_dry), name=c["name"]))


def handle(sc, body, name, length, radius, sleeve, material, mass, mu, z0=0.0):
    """A tool's handle from z0 to z0 + length along the body's z: the real round section (a capsule), or, in the
    `sleeves` variant, a square-section non-slip sleeve (a box of half width `sleeve`). Returns
    its volume share for the density bookkeeping (the handle is given `mass`)."""
    if sc.sleeves:
        sc.box(name, (0, 0, z0 + 0.5*length), (sleeve, sleeve, 0.5*length), material, body=body,
               density=mass/(4*sleeve*sleeve*length), mu=mu)
        return sleeve
    vol = math.pi*radius*radius*(length - 2*radius) + 4.0/3.0*math.pi*radius**3
    capsule(sc, body, name, (0, 0, z0 + radius), (0, 0, z0 + length - radius), radius, material, density=mass/vol, mu=mu,
            gap=P.MESH_CONTACT_GAP_M)
    return radius


def _box_collider(name, body, pos, half, cont_min, quat=(0, 0, 0, 1)):
    """A box for the continuum: no dimension under two grid cells (stated: a thinner blade sweeps up to a cell more)."""
    return dict(kind=CT.KIND_BOX, body=body, pos=tuple(pos), quat=tuple(quat), size=tuple(max(h, cont_min) for h in half),
                name=name)


def _capsule_collider(name, body, p0, p1, r, cont_min):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return dict(kind=CT.KIND_CAPSULE, body=body, pos=tuple(0.5*(p0 + p1)), quat=wq(rot_z_to(p1 - p0)),
                size=(max(r, cont_min), 0.5*float(np.linalg.norm(p1 - p0)), 0.), name=name)


class Whisk(HandTool):
    """Balloon whisk: handle, and the wire cage as one body sprung on two bending axes at the ferrule. The cage's
    stiffness at its tip is that of its wires bending together (params.WHISK); its envelope is the rigid collider;
    every wire is a chain of markers for the mixture."""
    kind = "whisk"
    CAGE = ((0.007, 0.0), (0.020, 0.035), (0.029, 0.080), (0.032, 0.112), (0.029, 0.145), (0.018, 0.166), (0.0, 0.175))

    def __init__(self, sc, name, pose, spec=None, cont_min=0.005):
        super().__init__()
        s = dict(P.WHISK, **(spec or {}))
        self.name, self.spec, self.length = name, s, s["length"]
        hl, hr, cl = s["handle_len"], s["handle_r"], s["cage_len"]
        scale_r, scale_z = s["cage_r"]/0.032, cl/0.175
        self.cage_profile = np.array([(r*scale_r, z*scale_z) for r, z in self.CAGE])
        m_cage = 0.3*s["mass"]
        H = link(sc, name, pose)
        g = handle(sc, H, f"{name}_handle", hl, hr, P.GRIP_SLEEVE["whisk"], "tool_handle", s["mass"] - m_cage, P.MU["default"])
        Cp = compose(pose, (0, 0, hl))
        C = link(sc, f"{name}_cage", Cp, mass=m_cage, com=(0, 0, 0.5*cl),
                 inertia=(m_cage*cl*cl/12, m_cage*cl*cl/12, m_cage*s["cage_r"]**2/2))
        cz = lambda z: float(np.interp(z, self.cage_profile[:, 1], self.cage_profile[:, 0]))
        capsule(sc, C, f"{name}_cage_neck", (0, 0, 0.004), (0, 0, 0.030*scale_z), 0.008, "collision_hidden", density=0.,
                mu=P.MU["steel_steel"], gap=P.MESH_CONTACT_GAP_M)
        sc.visual.add(sc.b.shape_count - 1)
        for kk, zf in enumerate((0.035, 0.078, 0.112, 0.146)):
            z = zf*scale_z
            r = min(cz(z), cl - z)
            sphere(sc, C, f"{name}_cage_s{kk}", (0, 0, z), r, "collision_hidden", density=0., mu=P.MU["steel_steel"],
                   gap=P.MESH_CONTACT_GAP_M)
            sc.visual.add(sc.b.shape_count - 1)
        k = s["tip_stiffness"]*cl*cl
        kd = 2*0.3*math.sqrt(k*m_cage*cl*cl/3)
        dof = lambda ax: sc.newton.ModelBuilder.JointDofConfig(
            axis=ax, limit_lower=-0.6, limit_upper=0.6, target_ke=k, target_kd=kd, target_pos=0., armature=1e-5,
            actuator_mode=sc.newton.JointTargetMode.POSITION)
        jf = sc.b.add_joint_free(H, label=f"{name}_free")
        jc = sc.b.add_joint_d6(H, C, angular_axes=[dof((1., 0., 0.)), dof((0., 1., 0.))],
                               parent_xform=rel_xform(sc, pose, Cp), child_xform=sc.wp.transform_identity(),
                               label=f"{name}_cage_bend")
        sc.b.add_articulation([jf, jc], label=f"{name}_assembly")
        self.body, self.cage = H, C
        self.bodies = [H, C]
        self.j = dict(bend=JointRef(jc, f"{name}_cage_bend"))
        self.grasp = dict(body=H, handle_len=hl, half=g)
        self.tip = (C, (0., 0., cl))
        # wires: each loop is two legs in a plane through the axis
        zs = np.linspace(0, cl, 60)
        rs = np.interp(zs, self.cage_profile[:, 1], self.cage_profile[:, 0])
        path = np.column_stack([rs, zs])
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s_len = np.concatenate([[0.], np.cumsum(seg)])
        n = max(4, int(round(s_len[-1]/0.005)))
        mids = (np.arange(n) + 0.5)*s_len[-1]/n
        mr, mz = np.interp(mids, s_len, path[:, 0]), np.interp(mids, s_len, path[:, 1])
        dr, dz = np.gradient(mr, mids), np.gradient(mz, mids)
        xs, ts = [], []
        for kk in range(2*int(s["loops"])):
            a = math.pi*kk/int(s["loops"])
            ca, sa = math.cos(a), math.sin(a)
            xs.append(np.column_stack([mr*ca, mr*sa, mz]))
            t = np.column_stack([dr*ca, dr*sa, dz])
            ts.append(t/np.linalg.norm(t, axis=1, keepdims=True))
        x = np.vstack(xs)
        self.markers = [dict(body=C, x=x, t=np.vstack(ts), length=np.full(len(x), s_len[-1]/n), dia=np.full(len(x), s["wire"]),
                             run=0.5*cl)]      # a film drips down half the cage's length
        self.colliders = [_capsule_collider(f"{name}_handle", H, (0, 0, hr), (0, 0, hl - hr), hr, cont_min)]
        self.wire_length = float(2*int(s["loops"])*s_len[-1])


class Spatula(HandTool):
    """Silicone spatula: handle with the blade's root, and the blade's tip half on a sprung hinge across its width."""
    kind = "spatula"

    def __init__(self, sc, name, pose, cont_min=0.0025):
        super().__init__()
        s = P.SPATULA
        self.name, self.length = name, s["length"]
        bw, bl = s["blade"]
        t, hr = s["blade_t"], s["handle_r"]
        hl = s["length"] - bl
        m_tip = 0.015
        H = link(sc, name, pose)
        # the handle and the blade's root share the mass by volume (one silicone-over-nylon moulding)
        v_root = bw*t*0.5*bl
        v_handle = (4*P.GRIP_SLEEVE["spatula"]**2*hl) if sc.sleeves else math.pi*hr*hr*hl
        dens = (s["mass"] - m_tip)/(v_handle + v_root)
        g = handle(sc, H, f"{name}_handle", hl, hr, P.GRIP_SLEEVE["spatula"], "silicone_red", dens*v_handle, P.MU["default"])
        sc.box(f"{name}_root", (0, 0, hl + 0.25*bl), (0.5*bw, 0.5*t, 0.25*bl), "silicone_red", body=H, density=dens,
               mu=P.MU["silicone"]*0.5)
        Tp = compose(pose, (0, 0, hl + 0.5*bl))
        T = link(sc, f"{name}_tip", Tp)
        sc.box(f"{name}_tip_blade", (0, 0, 0.25*bl), (0.5*bw, 0.4*t, 0.25*bl), "silicone_red", body=T,
               density=m_tip/(bw*0.8*t*0.5*bl), mu=P.MU["silicone"]*0.5)
        k = s["tip_stiffness"]*(0.5*bl)**2
        kd = 2*0.5*math.sqrt(k*m_tip*(0.5*bl)**2/3)
        jf = sc.b.add_joint_free(H, label=f"{name}_free")
        jt = sc.b.add_joint_revolute(H, T, parent_xform=rel_xform(sc, pose, Tp), child_xform=sc.wp.transform_identity(),
                                     axis=(1., 0., 0.), limit_lower=-0.9, limit_upper=0.9, target_ke=k, target_kd=kd,
                                     target_pos=0., armature=1e-6, actuator_mode=sc.newton.JointTargetMode.POSITION,
                                     label=f"{name}_blade_bend")
        sc.b.add_articulation([jf, jt], label=f"{name}_assembly")
        self.body, self.blade = H, T
        self.bodies = [H, T]
        self.j = dict(bend=JointRef(jt, f"{name}_blade_bend"))
        self.grasp = dict(body=H, handle_len=hl, half=g)
        self.tip = (T, (0., 0., 0.5*bl))
        self.edge_w = bw                           # the working edge that scrapes a pan's bottom (heat.Hob.scrape)
        self.colliders = [_capsule_collider(f"{name}_handle", H, (0, 0, hr), (0, 0, hl - hr), hr, cont_min),
                          _box_collider(f"{name}_root", H, (0, 0, hl + 0.25*bl), (0.5*bw, 0.5*t, 0.25*bl), cont_min),
                          _box_collider(f"{name}_tip", T, (0, 0, 0.25*bl), (0.5*bw, 0.5*t, 0.25*bl), cont_min)]


def dish_boxes(bw, bl, t, depth, rim):
    """A spoon's bowl as boxes: the octagon whose flats touch the oval bw x bl, as a plate t - depth thick with a lip
    `rim` wide standing `depth` above it along every flat. Bowl frame: x across, y out of the hollow, z along the tool,
    origin at the middle of the outline and of the thickness. Returns [(tag, centre, half extents, angle about y,
    share)]: a box's own x lies along its flat, its z points out of the bowl. The plate's boxes overlap one another
    and the lip's overlap at the corners; `share` is the part of a box's volume that counts as wood, so that the plate
    and the lip each weigh what their solid does (octagon and ring areas in closed form)."""
    a, b, k = 0.5*bw, 0.5*bl, math.tan(math.pi/8)
    tf = t - depth
    yf, yr = -0.5*depth, 0.5*tf
    V = np.array([(a, -b*k), (a, b*k), (a*k, b), (-a*k, b), (-a, b*k), (-a, -b*k), (-a*k, -b), (a*k, -b)])   # (x, z)
    E = np.roll(V, -1, axis=0) - V
    L = np.linalg.norm(E, axis=1)
    N = np.column_stack([E[:, 1], -E[:, 0]])/L[:, None]              # outward normals
    mid = V + 0.5*E
    ang = np.arctan2(N[:, 0], N[:, 1])
    area = 8*k*a*b
    turn = np.arccos(np.clip(np.sum(E*np.roll(E, -1, axis=0), axis=1)/(L*np.roll(L, -1)), -1., 1.))
    ring = L.sum()*rim - rim*rim*float(np.tan(0.5*turn).sum())      # the octagon less the one `rim` inside it
    ht = a*b*(1 - k)**2/L[1]                                         # height of the corner the two straight plates leave
    plate = [("blade", (0., yf, 0.), (a, 0.5*tf, b*k), 0.), ("blade_z", (0., yf, 0.), (a*k, 0.5*tf, b), 0.)]
    plate += [(f"blade_c{i//2}", (mid[i, 0] - 0.5*ht*N[i, 0], yf, mid[i, 1] - 0.5*ht*N[i, 1]), (0.5*L[i], 0.5*tf, 0.5*ht),
               float(ang[i])) for i in (1, 3, 5, 7)]
    lip = [(f"rim{i}", (mid[i, 0] - 0.5*rim*N[i, 0], yr, mid[i, 1] - 0.5*rim*N[i, 1]), (0.5*L[i], 0.5*depth, 0.5*rim),
            float(ang[i])) for i in range(8)]
    vol = lambda group: sum(8*h[0]*h[1]*h[2] for _, _, h, _ in group)
    return [(tag, tuple(map(float, c)), tuple(map(float, h)), a, float(share))
            for group, share in ((plate, area*tf/vol(plate)), (lip, ring*depth/vol(lip))) for tag, c, h, a in group]


class Paddle(HandTool):
    """A rigid spoon or flat stirrer: a round handle and its working end, a flat slab or, where the spec gives a hollow
    (bowl_depth: the wooden spoon), a dish of boxes (dish_boxes) with the hollow toward the tool's +y. The mixture
    meets the same boxes as the rigid world. (A flat slab cannot hold a spoonful.)"""

    def __init__(self, sc, name, pose, spec, kind, material, mu, cont_min=0.0025):
        super().__init__()
        self.kind, self.name, self.length = kind, name, spec["length"]
        bw, bl = spec["bowl"] if "bowl" in spec else spec["blade"]
        t = spec.get("bowl_t", spec.get("blade_t"))
        hr = spec["handle_r"]
        hl = spec["length"] - bl
        g = P.GRIP_SLEEVE[kind]
        grip_len = min(hl, 0.11)
        zc = hl + 0.5*bl
        if "bowl_depth" in spec:
            end = [(f"{name}_{tag}", (c[0], c[1], zc + c[2]), half, wq(Rotation.from_euler("y", a)), share)
                   for tag, c, half, a, share in dish_boxes(bw, bl, t, spec["bowl_depth"], spec["bowl_rim"])]
            v_end = sum(8*h[0]*h[1]*h[2]*share for _, _, h, _, share in end)
        else:
            end, v_end = [(f"{name}_blade", (0, 0, zc), (0.5*bw, 0.5*t, 0.5*bl), None, 1.)], bw*t*bl
        H = link(sc, name, pose)
        if sc.sleeves:
            vol = math.pi*hr*hr*(hl - grip_len) + 4*g*g*grip_len + v_end
            dens = spec["mass"]/vol
            sc.box(f"{name}_grip", (0, 0, 0.5*grip_len), (g, g, 0.5*grip_len), material, body=H, density=dens, mu=mu)
            if hl - grip_len > 2*hr:
                capsule(sc, H, f"{name}_handle", (0, 0, grip_len), (0, 0, hl), hr, material, density=dens, mu=mu,
                        gap=P.MESH_CONTACT_GAP_M)
        else:
            # the round shank it is sold with, from the butt to the blade
            dens = spec["mass"]/(math.pi*hr*hr*hl + v_end)
            g = handle(sc, H, f"{name}_handle", hl, hr, g, material, dens*math.pi*hr*hr*hl, mu)
        for lab, c, half, q, share in end:
            sc.box(lab, c, half, material, body=H, rot=q, density=dens*share, mu=mu)
        jf = sc.b.add_joint_free(H, label=f"{name}_free")
        sc.b.add_articulation([jf], label=f"{name}_assembly")
        self.body = H
        self.bodies = [H]
        self.grasp = dict(body=H, handle_len=grip_len, half=g)
        self.tip = (H, (0., 0., spec["length"]))
        self.edge_w = bw
        self.colliders = [_capsule_collider(f"{name}_handle", H, (0, 0, hr), (0, 0, hl), hr, cont_min)] + [
            _box_collider(lab, H, c, half, cont_min, quat=q or (0, 0, 0, 1)) for lab, c, half, q, _ in end]


class Thermometer(HandTool):
    """Pen-type digital probe thermometer: a round body and a thin stem. Its reading follows the temperature of the
    food round its tip with the tip's time constant (params.THERMOMETER; in air it drifts to the room's far more
    slowly). The reading is the instrument's output: the hob tasks give it to the policy as an observation, as a
    thermometer's display or its cable does. The mixture does not feel the 3.5 mm stem."""
    kind = "thermometer"

    def __init__(self, sc, name, pose, cont_min=0.0025):
        super().__init__()
        s = P.THERMOMETER
        self.name, self.length, self.spec = name, s["length"], s
        hl, hr, sr = s["handle_len"], s["handle_r"], s["stem_r"]
        H = link(sc, name, pose)
        g = handle(sc, H, f"{name}_handle", hl, hr, 0.007, "thermo_body", 0.8*s["mass"], P.MU["default"])
        stem_len = s["length"] - hl
        capsule(sc, H, f"{name}_stem", (0, 0, hl), (0, 0, s["length"] - 0.002), 0.002, "steel_wire",
                density=0.2*s["mass"]/(math.pi*0.002**2*stem_len), mu=P.MU["steel_steel"], gap=P.MESH_CONTACT_GAP_M)
        jf = sc.b.add_joint_free(H, label=f"{name}_free")
        sc.b.add_articulation([jf], label=f"{name}_assembly")
        self.body = H
        self.bodies = [H]
        self.grasp = dict(body=H, handle_len=hl, half=g)
        self.tip = (H, (0., 0., s["length"]))
        self.colliders = [_capsule_collider(f"{name}_handle", H, (0, 0, hr), (0, 0, hl - hr), hr, cont_min)]
        self.s_reading, self.s_in_food = 20.0, False
        self._out = None

    def post_step(self, w):
        if w.tick % 10:
            return
        s = self.spec
        target, tau, self.s_in_food = 20.0, s["tau_air_s"], False
        cont = w.cont
        if cont is not None and cont.n:
            wp = w.wp
            if self._out is None:
                self._out = wp.zeros(2, dtype=float, device=cont.device)
            q = w.bq[self.body]
            tip = q[:3] + Rotation.from_quat(q[3:]).apply(self.tip[1])
            self._out.zero_()
            wp.launch(CT.k_probe, dim=cont.capacity, inputs=[cont.x, cont.flags, cont.temp, cont.heatcap,
                                                             wp.vec3(*map(float, tip)), float(s["sense_r"]), self._out],
                      device=cont.device)
            c, ct = (float(v) for v in self._out.numpy())
            if c > 1e-6:
                target, tau, self.s_in_food = ct/c, s["tau_s"], True
        self.s_reading += (target - self.s_reading)*(1.0 - math.exp(-10.0*w.dt/tau))

    def state(self):
        return dict(reading_c=round(float(self.s_reading), 1), tip_in_food=bool(self.s_in_food))


# ====================================================================== hand mixer
class Mixer(HandTool):
    """Hand mixer. Frame: origin at the housing centre, x toward the beater end, z toward the handle. It rests on its
    heel (the rear face, -x down: how it is sold and set down, scene.py) and works with the beaters pointing down (-z).
    The handle is a round bar.

    The slide switch has OFF and five detents; the setting it rests in sets the free speed of the motor, whose torque
    falls on a straight line from stall at rest to zero at the free speed (a velocity drive of gain stall/free speed,
    integrated implicitly). The two beaters turn in opposite directions."""
    kind = "mixer"

    def __init__(self, sc, name, pose):
        super().__init__()
        s = P.MIXER
        self.name, self.spec = name, s
        hx, hy, hz = (0.5*v for v in s["body"])
        m_beater = 0.04
        m_house = s["mass"] - 2*m_beater - 0.01
        H = link(sc, name, pose)
        dens = m_house/(8*hx*hy*hz)
        sc.box(f"{name}_housing", (0, 0, 0), (hx, hy, hz), "mixer_white", body=H, density=dens, mu=P.MU["default"])
        hz_handle = hz + 0.045
        capsule(sc, H, f"{name}_handle", (-0.072, 0, hz_handle), (0.062, 0, hz_handle), s["handle_r"], "mixer_white",
                density=0., mu=P.MU["default"], gap=P.MESH_CONTACT_GAP_M)
        sc.box(f"{name}_strut_rear", (-0.072, 0, hz + 0.022), (0.012, 0.012, 0.024), "mixer_white", body=H, density=0.)
        sc.box(f"{name}_strut_front", (0.062, 0, hz + 0.022), (0.012, 0.012, 0.024), "mixer_white", body=H, density=0.)
        I = sc.wp.transform_identity()
        joints = [sc.b.add_joint_free(H, label=f"{name}_free")]
        # slide switch on the front strut's top, travelling toward the beater end
        self.detents = tuple(s["detent_pitch"]*k for k in range(6))
        Sp = compose(pose, (0.050, 0, hz_handle + s["handle_r"] + 0.004))
        S = link(sc, f"{name}_switch", Sp, mass=0.004, inertia=(2e-7, 2e-7, 2e-7))
        sc.box(f"{name}_switch_tab", (0, 0, 0), (0.006, 0.009, 0.004), "mixer_grey", body=S, density=0., mu=P.MU["pad"])
        ke = 2*s["detent_force"]/s["detent_pitch"]
        js = sc.b.add_joint_prismatic(H, S, parent_xform=rel_xform(sc, pose, Sp), child_xform=I, axis=(1., 0., 0.),
                                      limit_lower=-0.0005, limit_upper=self.detents[-1] + 0.0005, target_pos=0.,
                                      target_ke=ke, target_kd=2*0.7*math.sqrt(ke*0.012), armature=0.008,
                                      actuator_mode=sc.newton.JointTargetMode.POSITION, label=f"{name}_switch")
        joints.append(js)
        self.j = dict(switch=JointRef(js, f"{name}_switch"))
        self.beaters = []
        z_top = -hz
        zc = z_top - s["shaft_len"] - 0.5*s["beater_len"]
        for k, sign in enumerate((+1, -1)):
            y = sign*0.5*s["beater_gap"]
            Bp = compose(pose, (0.045, y, z_top))
            Bk = link(sc, f"{name}_beater{k}", Bp, mass=m_beater, com=(0, 0, -s["shaft_len"] - 0.5*s["beater_len"]),
                      inertia=(m_beater*0.03**2, m_beater*0.03**2, m_beater*s["beater_r"]**2*0.8))
            capsule(sc, Bk, f"{name}_shaft{k}", (0, 0, -0.004), (0, 0, -s["shaft_len"]), 0.003, "steel_wire", density=0.,
                    mu=P.MU["steel_steel"], gap=P.MESH_CONTACT_GAP_M)
            rc = 0.5*s["beater_gap"] - 0.0005
            capsule(sc, Bk, f"{name}_cage{k}", (0, 0, -s["shaft_len"] - rc), (0, 0, -s["shaft_len"] - s["beater_len"] + rc), rc,
                    "collision_hidden", density=0., mu=P.MU["steel_steel"], gap=P.MESH_CONTACT_GAP_M)
            sc.visual.add(sc.b.shape_count - 1)
            jb = sc.b.add_joint_revolute(H, Bk, parent_xform=rel_xform(sc, pose, Bp), child_xform=I, axis=(0., 0., 1.),
                                         target_ke=0., target_kd=0., effort_limit=s["stall_nm"]/2,
                                         actuator_mode=sc.newton.JointTargetMode.VELOCITY, armature=2e-5,
                                         label=f"{name}_beater{k}")
            joints.append(jb)
            self.j[f"beater{k}"] = JointRef(jb, f"{name}_beater{k}")
            self.beaters.append(Bk)
            # strips: `strips` flat wires per beater, each a bowed line from the shaft end to the tip
            n = 12
            zz = -s["shaft_len"] - (np.arange(n) + 0.5)*s["beater_len"]/n
            u = (np.arange(n) + 0.5)/n
            rr = s["beater_r"]*np.sin(math.pi*np.clip(0.08 + 0.84*u, 0, 1))**0.6
            for q in range(int(s["strips"])):
                a = 2*math.pi*q/int(s["strips"])
                x = np.column_stack([rr*math.cos(a), rr*math.sin(a), zz])
                t = np.tile([0., 0., -1.], (n, 1))
                self.markers.append(dict(body=Bk, x=x, t=t, length=np.full(n, s["beater_len"]/n*1.15),
                                         dia=np.full(n, s["strip"]), run=0.5*s["beater_len"]))
        sc.b.add_articulation(joints, label=f"{name}_assembly")
        self.body = H
        self.bodies = [H, S] + self.beaters
        self.switch_body = S
        self.grasp = dict(body=H, handle_centre=(-0.005, 0., hz_handle), handle_axis=(1., 0., 0.), handle_r=s["handle_r"])
        self.colliders = [_box_collider(f"{name}_housing", H, (0, 0, 0), (hx, hy, hz), 0.005)]
        self.s_setting = 0
        self.heel = -hx

    def setting(self):
        q = self.j["switch"].pos(self.w)
        return int(np.argmin(np.abs(np.asarray(self.detents) - q)))

    def pre_step(self, w):
        s = self.spec
        js = self.j["switch"]
        k = self.setting()
        self.s_setting = k
        w.tq[js.q] = self.detents[k]                      # the detent the slide is nearest to holds it
        free = s["speeds_rpm"][k]*RPM
        mjm = w.solver.mj_model
        for i, sign in enumerate((+1, -1)):
            jb = self.j[f"beater{i}"]
            gain = (0.5*s["stall_nm"]/free) if free > 0 else 0.02     # switched off: the gear train's drag only
            mjm.actuator_gainprm[jb.act][0] = gain
            mjm.actuator_biasprm[jb.act][2] = -gain
            w.tqd[jb.dof] = sign*free

    def state(self):
        w = self.w
        return dict(setting=int(self.s_setting), switch_m=round(self.j["switch"].pos(w), 5),
                    beater_rpm=[round(self.j[f"beater{i}"].vel(w)/RPM, 1) for i in range(2)],
                    free_rpm=float(self.spec["speeds_rpm"][self.s_setting]))


class HobKnob(HandTool):
    """The hob's rotary knob: off and the hob's twenty power levels, a detent every 15 degrees; the detent it rests in
    sets the power (params.HOB levels_w)."""
    kind = "knob"

    def __init__(self, sc, name, pos, level=0):
        super().__init__()
        h = P.HOB
        self.name = name
        pose = (np.asarray(pos, float), Rotation.identity())
        K = link(sc, name, pose, mass=0.06, inertia=(2e-5, 2e-5, 2e-5))
        # The cap stands 1.5 mm clear of the counter, as a knob on its spindle does. Flush with it the cap's
        # underside and the counter are in contact over its whole face: the knob does not rest in its detents,
        # wanders by tens of degrees under a torque of half the detent's, and a hand that holds and turns it meets
        # hundreds of newtons.
        gap = 0.0015
        disc(sc, K, f"{name}_cap", (0, 0, gap + 0.5*(h["knob_h"] - gap)), h["knob_r"], 0.5*(h["knob_h"] - gap), "knob_black",
             density=0., mu=P.MU["pad"])
        sc.box(f"{name}_grip", (0, 0, h["knob_h"] + 0.007), (0.020, 0.005, 0.007), "knob_black", body=K, density=0.,
               mu=P.MU["pad"])
        pitch = math.radians(h["detent_deg"])
        self.detents = tuple(pitch*k for k in range(len(h["levels_w"])))
        ke = 2*h["detent_nm"]/pitch
        jk = sc.b.add_joint_revolute(-1, K, parent_xform=sc.wp.transform(tuple(pose[0]), wq(pose[1])),
                                     child_xform=sc.wp.transform_identity(), axis=(0., 0., -1.), limit_lower=-0.02,
                                     limit_upper=self.detents[-1] + 0.02, target_pos=self.detents[level], target_ke=ke,
                                     target_kd=2*0.7*math.sqrt(ke*1e-4), armature=8e-5,
                                     actuator_mode=sc.newton.JointTargetMode.POSITION, label=name)
        sc.b.add_articulation([jk], label=f"{name}_assembly")
        sc.b.joint_q[sc.b.joint_q_start[jk]] = float(self.detents[level])
        self.body = K
        self.bodies = [K]
        self.j = dict(knob=JointRef(jk, name))
        self.s_level = int(level)

    def level(self):
        q = self.j["knob"].pos(self.w)
        return int(np.argmin(np.abs(np.asarray(self.detents) - q)))

    def pre_step(self, w):
        self.s_level = self.level()
        w.tq[self.j["knob"].q] = self.detents[self.s_level]

    def state(self):
        return dict(level=int(self.s_level), angle_deg=round(math.degrees(self.j["knob"].pos(self.w)), 2),
                    power_w=float(P.HOB["levels_w"][self.s_level]))

"""Actual FR3 rigid bodies and mass-bearing deformable microfiber meshes.

Only initial scene construction sets positions. Runtime has no grasp constraints.
"""
import hashlib
import math
import xml.etree.ElementTree as ET
import numpy as np
import newton
import newton.utils
import warp as wp
from newton.solvers import SolverMuJoCo, SolverStyle3D, SolverVBD, style3d

HOME = (0., -.4, 0., -2.2, 0., 1.9, .79)


def folded_grid(c, center, folded=True):
    n = round(c.cloth_side/c.cloth_edge)
    us = np.linspace(0,c.cloth_side,n+1)
    r = .004
    start = (c.cloth_side-math.pi*r)/2
    if folded:
        # Resolve the fold at ~1.6 mm, retaining the 10 mm maximum spacing on
        # flat regions. Coarse chord lengths otherwise preload a compressed fold.
        side_steps = math.ceil(start/c.cloth_edge)
        us = np.r_[np.linspace(0,start,side_steps+1),
                   np.linspace(start,start+math.pi*r,9)[1:],
                   np.linspace(start+math.pi*r,c.cloth_side,side_steps+1)[1:]]
    u, v = np.meshgrid(us,np.linspace(0,c.cloth_side,n+1),indexing="ij")
    uv = np.column_stack((u.ravel(), v.ravel()))
    p = np.column_stack((u.ravel(), v.ravel(), np.zeros(u.size)))
    if folded:
        # A semicircular fold with finite thickness, parameterized by rest arc
        # length. No coincident layers and no shortened material coordinates.
        s = u.ravel()-start
        theta = np.clip(s/r, 0, math.pi)
        p[:, 0] = np.where(s < 0, u.ravel(),
                          np.where(s <= math.pi*r, start+r*np.sin(theta),
                                   start-(s-math.pi*r)))
        p[:, 2] = r*(1-np.cos(theta))
    p[:, :2] -= (p[:, :2].min(axis=0)+p[:, :2].max(axis=0))/2
    p += np.asarray(center)
    triangles = []
    for i in range(len(us)-1):
        for j in range(n):
            a = i*(n+1)+j
            triangles.extend(((a, a+n+1, a+1), (a+1, a+n+1, a+n+2)))
    return uv, p, np.asarray(triangles, np.int32)


class Assembly:
    def __init__(self, c, *, robots=True, folded=True, render_only=False):
        self.config = c
        self.render_only = render_only
        newton.use_coord_layout_targets = True
        b = self.builder = newton.ModelBuilder(gravity=(0, 0, -9.80665))
        SolverMuJoCo.register_custom_attributes(b)
        SolverStyle3D.register_custom_attributes(b)
        SolverVBD.register_custom_attributes(b)
        b.default_shape_cfg = self.shape_config()
        self.sources, self.materials, self.objects = {}, {}, {}
        self.robot_joints, self.robot_bases, self.robot_yaws = {}, {}, {}
        self.cloths = []
        self.surface_shapes = {}
        self.build_world()
        centers = [(0, -.25, c.table_height+.010),
                   (0, .25, c.table_height+.010),
                   (-.27, .25, c.table_height+.010)]
        if c.scene=='kitchen':
            from .kitchen import build
            centers=build(self)
        if robots:
            self.add_robot("left", (0., -.78, c.table_height), math.pi/2)
            self.add_robot("right", (0., .78, c.table_height), -math.pi/2)
            if c.wrist_cameras:
                from control_cable.cameras import install
                install(self)
        for i in range(c.cloth_count):
            self.add_cloth(i, centers[i], folded)
        if c.cloth_solver=='vbd':
            # Explicit constitutive mapping, not an unnoticed engine default:
            # isotropic membrane shear/area stiffness and flat-rest bending.
            b.tri_materials=[(c.membrane_shear,c.membrane_lambda,c.membrane_damping,0.,0.) for _ in b.tri_materials]
            b.edge_rest_angle=[0. for _ in b.edge_rest_angle]
            uv_all=np.concatenate([cloth['uv'] for cloth in self.cloths])
            for i,(a,d,e,f) in enumerate(b.edge_indices):
                if a<0 or d<0:continue
                length=np.linalg.norm(uv_all[e]-uv_all[f])
                ea,fa=uv_all[e]-uv_all[a],uv_all[f]-uv_all[a]
                ed,fd=uv_all[e]-uv_all[d],uv_all[f]-uv_all[d]
                cross1=ea[0]*fa[1]-ea[1]*fa[0]
                cross2=ed[0]*fd[1]-ed[1]*fd[0]
                area=.5*(abs(float(cross1))+abs(float(cross2)))
                stiffness=2e-5*length/(2*area)
                b.edge_bending_properties[i]=(stiffness,stiffness*.01)
        b.color()

    def shape_config(self, **kwargs):
        values = dict(ke=20000., kd=100., mu=.45, gap=.001, margin=0., density=1000.,
                      mu_torsional=0., mu_rolling=0.)
        values.update(kwargs)
        return newton.ModelBuilder.ShapeConfig(**values)

    def box(self, name, center, half, material="steel", body=-1, cfg=None):
        idx = self.builder.add_shape_box(body, xform=wp.transform(center, wp.quat_identity()),
            hx=half[0], hy=half[1], hz=half[2], cfg=cfg or self.shape_config(), label=name)
        self.materials[idx] = material
        return idx

    def build_world(self):
        c, b = self.config, self.builder
        z = c.table_height
        self.surface_shapes["worktop"] = self.box("worktop", (0,0,z-.020),
            (c.table_depth/2,c.table_width/2,.020))
        self.surface_shapes["backsplash"] = self.box("backsplash", (c.table_depth/2+.002,0,z+c.backsplash_height/2),
            (.002,c.table_width/2,c.backsplash_height/2))
        for x in (-.31,.31):
            for y in (-.54,.54):
                self.box(f"leg_{x}_{y}", (x,y,(z-.04)/2), (.025,.025,(z-.04)/2))
        self.box("lower_shelf", (0,0,.22), (.33,.56,.012))
        for y in (-.78,.78):
            self.box(f"robot_pedestal_{y}", (0,y,z/2-.025), (.12,.12,z/2-.025), "charcoal")
            self.box(f"robot_mount_{y}", (0,y,z-.025), (.16,.16,.025), "aluminum")
        b.add_ground_plane(cfg=self.shape_config(mu=.65), label="floor")
        self.materials[b.shape_count-1] = "floor"

    def add_robot(self, name, position, yaw):
        from robots.backends.newton import add_robot as build, mount
        handle=build(self.builder,'fr3','wiping',instance=name,placement=mount(position,yaw),
            shape_config=self.shape_config(),sdf=self.config.cloth_solver=='vbd' and not self.render_only,
            parameters={'finger_effort_limit':self.config.gripper_effort,'finger_velocity_limit':self.config.gripper_speed,
                        'finger_kp':self.config.gripper_stiffness})
        self.sources['fr3_urdf']=handle.source
        self.robot_joints.update(handle.metadata['legacy_addresses'])
        for joint_name,spec in handle.metadata['legacy_addresses'].items():
            self.builder.joint_limit_ke[spec['dof']]=self.config.joint_limit_ke
            self.builder.joint_limit_kd[spec['dof']]=self.config.joint_limit_kd
            if 'finger' in joint_name:
                self.builder.joint_target_kd[spec['dof']]=self.config.gripper_effort/self.config.gripper_speed
        self.robot_bases[name]=position;self.robot_yaws[name]=yaw
        for i in handle.shape_ids:self.materials[i]='robot_white'

    def add_cloth(self, index, center, folded):
        c,b = self.config,self.builder
        uv,p,tris = folded_grid(c, center, folded)
        first, first_tri = b.particle_count,b.tri_count
        style3d.add_cloth_mesh(b,pos=wp.vec3(0,0,0),rot=wp.quat_identity(),vel=wp.vec3(0,0,0),
            panel_verts=uv.tolist(), panel_indices=tris.ravel().tolist(),
            vertices=p.tolist(), indices=tris.ravel().tolist(),
            density=c.cloth_dry_mass/c.cloth_side**2,scale=1.,particle_radius=c.cloth_radius,
            tri_aniso_ke=wp.vec3(120.,120.,25.),edge_aniso_ke=wp.vec3(2e-5,2e-5,1e-5))
        self.cloths.append(dict(name=f"cloth_{index}",first=first,count=len(p),
            first_triangle=first_tri,triangles=tris,uv=uv,initial_water_fraction=c.initial_water_fraction if index<2 else 0.))

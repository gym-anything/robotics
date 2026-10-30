"""Native full-face/edge cloth contact plus the same explicit lagged reaction.

Uses Newton's own final-contact force law and material parameters. No virtual
grasp constraints or invented fingertip forces. MuJoCo remains the finite-effort
robot integrator: VBD alone does not support joint_effort_limit in Newton 1.5.
"""
from types import SimpleNamespace
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverVBD
from newton._src.solvers.vbd.rigid_vbd_kernels import _eval_soft_ef_contact
from newton._src.geometry.soft_contacts_sdf import eval_shape_sdf
from newton._src.geometry.sdf_texture import TextureSDFData


def material_neighborhood(solver,sim):
    """Exclude local panel neighbors, not distinct folded layers or cloths.

    Fold refinement has1.56mm edges inside a3mm shell. Topological rings alone
    leave thousands of repulsive edge contacts in an unstretched fold. Use
    Newton's rest-distance filter with FLAT material coordinates. Separate
    panels are offset in this reference space so cross-cloth contact survives.
    This changes collision candidate selection only, never physical positions.
    """
    radius=sim.config.self_contact_material_exclusion
    if not radius:return
    reference=np.zeros((sim.model.particle_count,3),np.float32)
    for i,cloth in enumerate(sim.assembly.cloths):
        ids=slice(cloth['first'],cloth['first']+cloth['count'])
        reference[ids,:2]=cloth['uv']
        reference[ids,0]+=i*(sim.config.cloth_side+1.)
    solver.particle_q_rest=wp.array(reference,dtype=wp.vec3,device=solver.device)
    solver.particle_rest_shape_contact_exclusion_radius=radius


@wp.kernel
def cloth_pair_material(count:wp.array[int],shape:wp.array[int],shape_mu:wp.array[float],
    k:float,d:float,mu:float,penalty:wp.array[float],maximum:wp.array[float],
    damping:wp.array[float],friction:wp.array[float]):
    i=wp.tid()
    if i>=count[0]:return
    # These are PAIR coefficients for soft microfiber against a rigid surface,
    # not a cloth constituent to average with the rigid-rigid contact stiffness.
    penalty[i]=k;maximum[i]=k;damping[i]=d
    friction[i]=wp.sqrt(mu*shape_mu[shape[i]])


@wp.kernel
def refresh_sphere_contacts(count:wp.array[int],indices:wp.array[wp.vec3i],bary:wp.array[wp.vec3],
    shape:wp.array[int],shape_type:wp.array[int],shape_body:wp.array[int],
    shape_transform:wp.array[wp.transform],scale:wp.array[wp.vec3],
    body_q:wp.array[wp.transform],particle_q:wp.array[wp.vec3],
    point:wp.array[wp.vec3],normal:wp.array[wp.vec3]):
    i=wp.tid()
    if i>=count[0]:return
    s=shape[i]
    if shape_type[s]!=int(newton.GeoType.SPHERE):return
    body=shape_body[s];pose=wp.transform_identity()
    if body>=0:pose=body_q[body]
    center=wp.transform_point(pose,wp.transform_get_translation(shape_transform[s]))
    x=wp.vec3(0.)
    for k in range(3):
        if indices[i][k]>=0:x+=bary[i][k]*particle_q[indices[i][k]]
    d=x-center;length=wp.length(d);n=normal[i]
    if length>1e-8:n=d/length
    # A sphere's occupied volume is rotationally invariant. Reusing a rotating
    # body-local surface anchor with a FROZEN world normal violates that: the
    # solver can reduce normal overlap by rotating the sphere without moving
    # its center. Refresh existing rows, not state, forces, stiffness or counts.
    point[i]=wp.transform_point(wp.transform_inverse(pose),center+scale[s][0]*n)
    normal[i]=n


@wp.kernel
def refresh_mesh_contacts(count:wp.array[int],indices:wp.array[wp.vec3i],bary:wp.array[wp.vec3],
    shape:wp.array[int],shape_type:wp.array[int],shape_body:wp.array[int],
    shape_transform:wp.array[wp.transform],scale:wp.array[wp.vec3],
    sdf_index:wp.array[int],sdf_table:wp.array[TextureSDFData],
    body_q:wp.array[wp.transform],particle_q:wp.array[wp.vec3],
    point:wp.array[wp.vec3],normal:wp.array[wp.vec3]):
    i=wp.tid()
    if i>=count[0]:return
    s=shape[i]
    if shape_type[s]!=int(newton.GeoType.MESH) or sdf_index[s]<0:return
    body=shape_body[s];pose=wp.transform_identity()
    if body>=0:pose=body_q[body]
    shape_pose=pose*shape_transform[s]
    x=wp.vec3(0.)
    for k in range(3):
        if indices[i][k]>=0:x+=bary[i][k]*particle_q[indices[i][k]]
    local=wp.transform_point(wp.transform_inverse(shape_pose),x)
    lower,phi,gradient=eval_shape_sdf(shape_type[s],scale[s],local,sdf_index[s],sdf_table)
    n=wp.transform_vector(shape_pose,gradient)
    if wp.length(n)>1e-8:
        n=wp.normalize(n)
        point[i]=wp.transform_point(wp.transform_inverse(pose),x-phi*n)
        normal[i]=n


@wp.kernel
def sphere_normal_velocity(count:wp.array[int],indices:wp.array[wp.vec3i],bary:wp.array[wp.vec3],
    shape:wp.array[int],shape_type:wp.array[int],shape_body:wp.array[int],
    shape_transform:wp.array[wp.transform],body_q:wp.array[wp.transform],
    body_prev:wp.array[wp.transform],particle_q:wp.array[wp.vec3],particle_prev:wp.array[wp.vec3],
    point:wp.array[wp.vec3],normal:wp.array[wp.vec3],dt:float,velocity:wp.array[wp.vec3]):
    i=wp.tid()
    if i>=count[0]:return
    s=shape[i]
    if shape_type[s]!=int(newton.GeoType.SPHERE):return
    body=shape_body[s];pose=wp.transform_identity();previous=pose
    if body>=0:pose=body_q[body];previous=body_prev[body]
    local_center=wp.transform_get_translation(shape_transform[s])
    center=wp.transform_point(pose,local_center);old_center=wp.transform_point(previous,local_center)
    x=wp.vec3(0.);old_x=wp.vec3(0.)
    for k in range(3):
        if indices[i][k]>=0:
            x+=bary[i][k]*particle_q[indices[i][k]]
            old_x+=bary[i][k]*particle_prev[indices[i][k]]
    # The native finite-difference surface anchor includes a radial chord
    # under pure rotation. That is NOT normal closing of a sphere's surface.
    # Encode the exact signed-distance change in its normal velocity channel;
    # preserve the native tangential slip and its Coulomb friction unchanged.
    displacement=wp.transform_point(pose,point[i])-wp.transform_point(previous,point[i])
    raw_normal=wp.dot(normal[i],x-old_x-displacement)
    geometric_normal=wp.length(x-center)-wp.length(old_x-old_center)
    correction=normal[i]*((raw_normal-geometric_normal)/dt)
    velocity[i]=wp.transform_vector(wp.transform_inverse(pose),correction)


@wp.kernel
def mass_scaled_seed(count:wp.array[int],shape0:wp.array[int],shape1:wp.array[int],
    shape_body:wp.array[int],inv_mass:wp.array[float],scale:float,
    maximum:wp.array[float],penalty:wp.array[float]):
    i=wp.tid()
    if i>=count[0]:return
    a=shape_body[shape0[i]];b=shape_body[shape1[i]];inverse=0.
    if a>=0:inverse+=inv_mass[a]
    if b>=0:inverse+=inv_mass[b]
    if inverse>0.:
        # Numerical AVBD seed, not a changed final material or force cap.
        # Preserve any larger history seed and the native material ceiling.
        seed=wp.min(maximum[i],wp.max(100.,scale/inverse))
        penalty[i]=wp.max(penalty[i],seed)


@wp.kernel
def mass_relative_damping(count:wp.array[int],shape0:wp.array[int],shape1:wp.array[int],
    shape_body:wp.array[int],inv_mass:wp.array[float],ratio:float,
    stiffness:wp.array[float],damping:wp.array[float]):
    i=wp.tid()
    if i>=count[0]:return
    a=shape_body[shape0[i]];b=shape_body[shape1[i]];inverse=0.
    if a>=0:inverse+=inv_mass[a]
    if b>=0:inverse+=inv_mass[b]
    if inverse>0.:
        # Explicit pair constitutive estimate, not velocity/force clipping.
        # A single100N*s/m dashpot overdamps0.1g grains by roughly35x critical.
        # Retain authored lower damping where it is already below this ratio.
        damping[i]=wp.min(damping[i],2.*ratio*wp.sqrt(stiffness[i]/inverse))


class ClothPairVBD(SolverVBD):
    def __init__(self,*args,contact_seed_mass_scale=0.,refresh_sphere_contact_geometry=False,
                 refresh_mesh_contact_geometry=False,
                 compression_core=False,self_contact_increments=False,exact_membrane_hessian=False,
                 rigid_contact_damping_ratio=0.,gap_aware_rigid_damping=False,zero_slip_hessian=False,
                 self_zero_slip_hessian=False,**kwargs):
        if refresh_sphere_contact_geometry and kwargs.get('integrate_with_external_rigid_solver',False):
            raise ValueError('Sphere refresh requires jointly solved native rigid/cloth state')
        self.contact_seed_mass_scale=contact_seed_mass_scale
        self.rigid_contact_damping_ratio=rigid_contact_damping_ratio
        self.refresh_sphere_contact_geometry=refresh_sphere_contact_geometry
        self.refresh_mesh_contact_geometry=refresh_mesh_contact_geometry
        self._gap_methods=None
        self._friction_methods=None
        self._core_methods=None
        self.record_contact_kernel=collect
        self.core_redirected_kernels=()
        if compression_core:
            if zero_slip_hessian or self_zero_slip_hessian:
                raise ValueError('Do not combine unqualified constitutive candidates')
            if np.any(np.abs(args[0].particle_radius.numpy()-.0015)>1e-8):
                raise ValueError('Compression candidate is defined for the1.5mm pile radius only')
            from .compression_core import redirected_methods
            self._core_methods,self.record_contact_kernel,self.core_redirected_kernels=redirected_methods()
        self.friction_redirected_kernels=()
        if self_zero_slip_hessian and not zero_slip_hessian:raise ValueError('Self-limit diagnostic requires body-limit correction')
        if zero_slip_hessian:
            if gap_aware_rigid_damping:raise ValueError('Do not combine unqualified contact experiments')
            from .friction_limit import redirected_methods
            self._friction_methods,self.friction_redirected_kernels=redirected_methods(include_self_contact=self_zero_slip_hessian)
        if gap_aware_rigid_damping:
            from .rigid_gap_damping import redirected_methods,redirect_rigid_method
            self._gap_methods=redirected_methods()
            if self._core_methods:
                self._core_methods=(redirect_rigid_method(self._core_methods[0]),*self._core_methods[1:])
        self._self_increment_method=None
        self.self_increment_kernels=()
        if self_contact_increments:
            if self._friction_methods:
                raise ValueError('Do not combine the increment candidate with unrelated friction/damping candidates')
            # The gap-onset correction touches rigid/rigid dispatch and its
            # force export only. The increment correction below wraps only
            # particle self-contact in the existing (possibly core) method.
            # They do not replace one another's functions or constitutive law.
            from .self_contact_increment import redirect_particle_method
            base=self._core_methods[1] if self._core_methods else SolverVBD._solve_particle_iteration
            self._self_increment_method,self.self_increment_kernels=redirect_particle_method(base)
        self._membrane_method=None
        self.membrane_hessian_kernels=()
        if exact_membrane_hessian:
            from .membrane_hessian import redirect_particle_method
            base=(self._self_increment_method or
                  (self._core_methods[1] if self._core_methods else
                   self._friction_methods[1] if self._friction_methods else SolverVBD._solve_particle_iteration))
            self._membrane_method,self.membrane_hessian_kernels=redirect_particle_method(base)
        super().__init__(*args,**kwargs)

    def _refresh_sphere_contacts(self,state,contacts,dt):
        if not self.refresh_sphere_contact_geometry or contacts is None or not self.model.particle_count:return
        m=self.model
        wp.launch(refresh_sphere_contacts,dim=contacts.soft_contact_max,inputs=[
            contacts.soft_contact_count,contacts.soft_contact_indices,contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,m.shape_type,m.shape_body,m.shape_transform,m.shape_scale,
            state.body_q,state.particle_q],outputs=[contacts.soft_contact_body_pos,contacts.soft_contact_normal],device=m.device)
        wp.launch(sphere_normal_velocity,dim=contacts.soft_contact_max,inputs=[
            contacts.soft_contact_count,contacts.soft_contact_indices,contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,m.shape_type,m.shape_body,m.shape_transform,
            state.body_q,self.body_q_prev,state.particle_q,self.particle_q_prev,
            contacts.soft_contact_body_pos,contacts.soft_contact_normal,dt],
            outputs=[contacts.soft_contact_body_vel],device=m.device)

    def _solve_particle_iteration(self,state_in,state_out,contacts,dt,iter_num):
        self._refresh_sphere_contacts(state_in,contacts,dt)
        self._refresh_mesh_contacts(state_in,contacts)
        if self._membrane_method:
            return self._membrane_method(self,state_in,state_out,contacts,dt,iter_num)
        if self._self_increment_method:
            return self._self_increment_method(self,state_in,state_out,contacts,dt,iter_num)
        if self._core_methods:
            return self._core_methods[1](self,state_in,state_out,contacts,dt,iter_num)
        if self._friction_methods:
            return self._friction_methods[1](self,state_in,state_out,contacts,dt,iter_num)
        return super()._solve_particle_iteration(state_in,state_out,contacts,dt,iter_num)

    def _solve_rigid_body_iteration(self,state_in,state_out,control,contacts,dt):
        self._refresh_sphere_contacts(state_in,contacts,dt)
        self._refresh_mesh_contacts(state_in,contacts)
        if self._core_methods:
            return self._core_methods[0](self,state_in,state_out,control,contacts,dt)
        if self._friction_methods:
            return self._friction_methods[0](self,state_in,state_out,control,contacts,dt)
        if self._gap_methods:
            return self._gap_methods[0](self,state_in,state_out,control,contacts,dt)
        return super()._solve_rigid_body_iteration(state_in,state_out,control,contacts,dt)

    def coupling_harvest_proxy_wrenches(self,*args,**kwargs):
        if self._core_methods:return self._core_methods[2](self,*args,**kwargs)
        return super().coupling_harvest_proxy_wrenches(*args,**kwargs)

    def _refresh_mesh_contacts(self,state,contacts):
        # Candidate only: refresh current geometric linearization of existing
        # rows. No new contact, barycentric weight, material, force, velocity or
        # state is prescribed here. This is not CCD or a penetration guarantee.
        if not self.refresh_mesh_contact_geometry or contacts is None:return
        if self.integrate_with_external_rigid_solver:
            raise ValueError('Mesh refresh requires jointly solved native rigid/cloth state')
        m=self.model
        wp.launch(refresh_mesh_contacts,dim=contacts.soft_contact_max,inputs=[
            contacts.soft_contact_count,contacts.soft_contact_indices,contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,m.shape_type,m.shape_body,m.shape_transform,m.shape_scale,
            m._shape_sdf_index,m._texture_sdf_data,
            state.body_q,state.particle_q],outputs=[contacts.soft_contact_body_pos,contacts.soft_contact_normal],device=m.device)

    def collect_rigid_contact_forces(self,*args,**kwargs):
        if self._gap_methods:return self._gap_methods[1](self,*args,**kwargs)
        return super().collect_rigid_contact_forces(*args,**kwargs)

    def _initialize_rigid_bodies(self,state_in,control,contacts,dt,refresh):
        super()._initialize_rigid_bodies(state_in,control,contacts,dt,refresh)
        if self._gap_methods and contacts is not None and self.rigid_contact_hard:
            from .rigid_gap_damping import remove_initial_air_gap
            wp.launch(remove_initial_air_gap,dim=contacts.rigid_contact_max,inputs=[
                contacts.rigid_contact_count,contacts.rigid_contact_normal,
                self.body_body_contact_C0],device=self.device)
        if contacts is not None and self.contact_seed_mass_scale>0.:
            wp.launch(mass_scaled_seed,dim=contacts.rigid_contact_max,inputs=[
                contacts.rigid_contact_count,contacts.rigid_contact_shape0,contacts.rigid_contact_shape1,
                self.model.shape_body,self.body_inv_mass_effective,self.contact_seed_mass_scale,
                self.body_body_contact_material_ke],outputs=[self.body_body_contact_penalty_k],device=self.device)
        if contacts is not None and self.rigid_contact_damping_ratio>0.:
            wp.launch(mass_relative_damping,dim=contacts.rigid_contact_max,inputs=[
                contacts.rigid_contact_count,contacts.rigid_contact_shape0,contacts.rigid_contact_shape1,
                self.model.shape_body,self.body_inv_mass_effective,self.rigid_contact_damping_ratio,
                self.body_body_contact_material_ke],outputs=[self.body_body_contact_material_kd],device=self.device)
        if contacts is not None and contacts.soft_contact_max>0:
            m=self.model
            wp.launch(cloth_pair_material,dim=contacts.soft_contact_max,inputs=[
                contacts.soft_contact_count,contacts.soft_contact_shape,m.shape_material_mu,
                m.soft_contact_ke,m.soft_contact_kd,m.soft_contact_mu],outputs=[
                self.body_particle_contact_penalty_k,self.body_particle_contact_material_ke,
                self.body_particle_contact_material_kd,self.body_particle_contact_material_mu],device=m.device)


@wp.kernel
def collect(dt:float,pos:wp.array[wp.vec3],prev:wp.array[wp.vec3],radius:wp.array[float],
    body_q:wp.array[wp.transform],body_prev:wp.array[wp.transform],body_qd:wp.array[wp.spatial_vector],
    com:wp.array[wp.vec3],eps:float,ke:wp.array[float],kd:wp.array[float],mu:wp.array[float],
    count:wp.array[int],indices:wp.array[wp.vec3i],bary:wp.array[wp.vec3],shape:wp.array[int],
    point:wp.array[wp.vec3],velocity:wp.array[wp.vec3],normal:wp.array[wp.vec3],
    margin:wp.array[float],shape_body:wp.array[int],
    wrenches:wp.array[wp.spatial_vector],forces:wp.array[wp.vec3]):
    i=wp.tid()
    if i>=wp.min(count[0],forces.shape[0]):return
    f,h,p=_eval_soft_ef_contact(i,indices[i],bary[i],pos,prev,radius,ke[i],kd[i],mu[i],eps,
        shape_body,body_q,body_prev,body_qd,com,shape,point,velocity,normal,margin,dt)
    forces[i]=f
    b=shape_body[shape[i]]
    if b>=0:
        center=wp.transform_point(body_q[b],com[b])
        wp.atomic_add(wrenches,b,wp.spatial_vector(-f,wp.cross(p-center,-f)))


class FullSurfaceCloth:
    def __init__(self,sim):
        m,c=sim.model,sim.config
        self.model=m
        self.device=m.device
        self.native=ClothPairVBD(m,iterations=c.nonlinear_iterations,
            self_contact_increments=c.native_self_contact_increments,
            zero_slip_hessian=c.native_zero_slip_hessian!='off',
            self_zero_slip_hessian=c.native_zero_slip_hessian=='both',
            friction_epsilon=c.friction_velocity,
            integrate_with_external_rigid_solver=True,particle_enable_self_contact=True,
            particle_self_contact_radius=2*c.cloth_radius,
            particle_self_contact_margin=4*c.cloth_radius,
            particle_topological_contact_filter_threshold=c.self_contact_filter_hops,
            particle_vertex_contact_buffer_size=64,particle_edge_contact_buffer_size=128,
            rigid_body_particle_contact_buffer_size=32768)
        material_neighborhood(self.native,sim)
        self.collision=SimpleNamespace(
            wrenches=wp.zeros(m.body_count,dtype=wp.spatial_vector,device=c.device),
            forces=wp.zeros(sim.contacts.soft_contact_max,dtype=wp.vec3,device=c.device))

    def step(self,state_in,state_out,control,contacts,dt):
        self.native.step(state_in,state_out,control,contacts,dt)
        m,s=self.model,self.native
        self.collision.wrenches.zero_();self.collision.forces.zero_()
        wp.launch(collect,dim=contacts.soft_contact_max,inputs=[dt,
            state_out.particle_q,s.particle_q_prev,m.particle_radius,
            state_out.body_q,state_in.body_q,state_out.body_qd,m.body_com,s.friction_epsilon,
            s.body_particle_contact_penalty_k,s.body_particle_contact_material_kd,s.body_particle_contact_material_mu,
            contacts.soft_contact_count,contacts.soft_contact_indices,contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,contacts.soft_contact_body_pos,contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,m.shape_margin,m.shape_body],
            outputs=[self.collision.wrenches,self.collision.forces],device=m.device)

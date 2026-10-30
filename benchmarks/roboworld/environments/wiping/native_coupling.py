"""Native finite-inertia rigid/cloth coupling.

MuJoCo owns finite-effort robots and, in production, every free rigid body so
robot/prop contacts share articulation constraints and hard stops. VBD owns the
cloth and receives finite-inertia proxies for cloth contact. Historical split
ownership remains an explicit diagnostic configuration.
"""
from types import SimpleNamespace
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupledProxy
from .vbd_contact import ClothPairVBD,collect,material_neighborhood


class RecordedPairVBD(ClothPairVBD):
    def __init__(self,model,*,rigid_iterations,rigid_phase='first',adaptive_initial_guess=False,local_frame_z=0.,
                 translation_refinement=False,cloth_count=1,**kwargs):
        self.rigid_iterations=rigid_iterations
        if rigid_phase not in ('first','last'):raise ValueError('Unknown rigid pass phase')
        self.rigid_phase=rigid_phase
        if kwargs['iterations']%rigid_iterations:raise ValueError('Cloth iterations must divide into exact rigid passes')
        self.rigid_stride=kwargs['iterations']//rigid_iterations
        super().__init__(model,**kwargs)
        self.adaptive_initial_guess=adaptive_initial_guess
        if adaptive_initial_guess:
            self.initial_guess_old_velocity=wp.clone(model.particle_qd)
            self.initial_guess_valid=wp.zeros(model.particle_count,dtype=int,device=model.device)
            self.initial_guess_factors=wp.zeros(model.particle_count,dtype=float,device=model.device)
        self.previous_for_record=wp.clone(model.body_q)
        self.record=SimpleNamespace(forces=wp.zeros(32768,dtype=wp.vec3,device=model.device),
            wrenches=wp.zeros(model.body_count,dtype=wp.spatial_vector,device=model.device))
        self.local_frame=None
        if local_frame_z:
            from .local_frame import LocalSolveFrame
            self.local_frame=LocalSolveFrame(self,local_frame_z)
        self.translation_refinement=None
        if translation_refinement:
            from .translation_refinement import TranslationRefinement
            self.translation_refinement=TranslationRefinement(self,cloth_count)

    def _solve_particle_iteration(self,state_in,state_out,contacts,dt,iter_num):
        super()._solve_particle_iteration(state_in,state_out,contacts,dt,iter_num)
        if self.translation_refinement:
            self.translation_refinement.refine(state_in,state_out,contacts,dt)

    def step(self,state_in,state_out,control,contacts,dt):
        if self.local_frame is None:
            return super().step(state_in,state_out,control,contacts,dt)
        with self.local_frame.scope():
            super().step(state_in,state_out,control,contacts,dt)
            self.local_frame.remember(state_out)

    def coupling_harvest_proxy_wrenches(self,*args,**kwargs):
        if self.local_frame is None:return super().coupling_harvest_proxy_wrenches(*args,**kwargs)
        # Feedback must evaluate the same precise geometry as the solve, not
        # recompute friction after rounding positions back to public floats.
        with self.local_frame.scope():
            return super().coupling_harvest_proxy_wrenches(*args,**kwargs)

    def _initialize_particles(self,state_in,state_out,dt):
        if self.adaptive_initial_guess:
            from .adaptive_initialization import initialize
            return initialize(self,state_in,state_out,dt)
        return super()._initialize_particles(state_in,state_out,dt)

    def _initialize_rigid_bodies(self,state_in,control,contacts,dt,refresh):
        self._iteration=0;self.active_contacts=contacts
        super()._initialize_rigid_bodies(state_in,control,contacts,dt,refresh)

    def _solve_rigid_body_iteration(self,*args,**kwargs):
        offset=0 if self.rigid_phase=='first' else self.rigid_stride-1
        execute=self._iteration%self.rigid_stride==offset;self._iteration+=1
        if execute:return super()._solve_rigid_body_iteration(*args,**kwargs)

    def _finalize_rigid_bodies(self,state_in,state_out,dt):
        wp.copy(self.previous_for_record,self.body_q_prev)
        super()._finalize_rigid_bodies(state_in,state_out,dt)

    def _finalize_particles(self,state_out,dt):
        super()._finalize_particles(state_out,dt)
        c=self.active_contacts;m=self.model
        self.record.forces.zero_();self.record.wrenches.zero_()
        if not m.particle_count:return
        wp.launch(self.record_contact_kernel,dim=c.soft_contact_max,inputs=[dt,state_out.particle_q,self.particle_q_prev,m.particle_radius,
            state_out.body_q,self.previous_for_record,state_out.body_qd,m.body_com,self.friction_epsilon,
            self.body_particle_contact_penalty_k,self.body_particle_contact_material_kd,self.body_particle_contact_material_mu,
            c.soft_contact_count,c.soft_contact_indices,c.soft_contact_barycentric,c.soft_contact_shape,
            c.soft_contact_body_pos,c.soft_contact_body_vel,c.soft_contact_normal,m.shape_margin,m.shape_body],
            outputs=[self.record.wrenches,self.record.forces],device=m.device)


@wp.kernel
def remap_soft(count:wp.array[int],shape:wp.array[int],indices:wp.array[wp.vec3i],
    shapes_to_global:wp.array[int],particles_to_global:wp.array[int],out_shape:wp.array[int],out_indices:wp.array[wp.vec3i]):
    i=wp.tid()
    if i>=count[0]:return
    out_shape[i]=shapes_to_global[shape[i]]
    ids=indices[i]
    for j in range(3):
        if ids[j]>=0:ids[j]=particles_to_global[ids[j]]
    out_indices[i]=ids


@wp.kernel
def remap_wrench(source:wp.array[wp.spatial_vector],mapping:wp.array[int],dest:wp.array[wp.spatial_vector]):
    i=wp.tid();dest[mapping[i]]=source[i]


class NativeCoupled:
    def __init__(self,sim):
        m,c=sim.model,sim.config;self.device=m.device;self.model=m
        if c.native_rigid_ownership=='mujoco':
            bodies=list(range(m.body_count))
        else:
            prefixes=('left/','right/','crumb_') if c.native_grain_solver=='mujoco' else ('left/','right/')
            bodies=[i for i,n in enumerate(m.body_label) if n.startswith(prefixes)]
        props=[i for i in range(m.body_count) if i not in bodies]
        joints=[j for j,b in enumerate(m.joint_child.numpy()) if b in bodies]
        prop_joints=[j for j,b in enumerate(m.joint_child.numpy()) if b in props]
        finger_joints=[j for j in joints if int(m.joint_type.numpy()[j])==int(newton.JointType.PRISMATIC)
            and 'finger' in m.body_label[int(m.joint_child.numpy()[j])]] if c.native_proxy_finger_joints else []
        def cloth_solver(v):
            if c.native_rigid_ownership=='mujoco':
                # Rigid contacts are already solved once, with articulation and
                # actuator constraints, by the source. The VBD view retains all
                # shapes only as finite-inertia cloth-contact proxies.
                flags=v._cow_array('shape_flags')
                values=flags.numpy()
                values&=~int(newton.ShapeFlags.COLLIDE_SHAPES)
                flags.assign(values)
            if finger_joints:
                # Preserve only the physical finger rails in the cloth-side
                # proxy. The finite-effort actuators remain MuJoCo-owned:
                # no duplicate/uncapped VBD drive may close these fingers.
                v.joint_target_ke=wp.zeros_like(v.joint_target_ke)
                v.joint_target_kd=wp.zeros_like(v.joint_target_kd)
            return RecordedPairVBD(v,
                iterations=c.nonlinear_iterations,rigid_iterations=c.native_rigid_iterations,
                rigid_phase=c.native_rigid_phase,
                adaptive_initial_guess=c.native_adaptive_initial_guess,
                local_frame_z=c.native_local_frame_z,
                translation_refinement=c.native_translation_refinement,cloth_count=c.cloth_count,
                rigid_contact_history=c.rigid_contact_history,
                rigid_avbd_linear_beta=c.native_rigid_linear_beta,
                rigid_avbd_contact_alpha=c.native_rigid_contact_alpha,
                contact_seed_mass_scale=c.native_contact_seed_mass_scale,
                gap_aware_rigid_damping=c.native_rigid_contact_onset_correction,
                refresh_sphere_contact_geometry=c.native_sphere_contact_refresh,
                refresh_mesh_contact_geometry=c.native_mesh_contact_refresh,
                compression_core=c.native_contact_compression_core,
                self_contact_increments=c.native_self_contact_increments,
                rigid_contact_damping_ratio=c.native_rigid_contact_damping_ratio,
                zero_slip_hessian=c.native_zero_slip_hessian!='off',
                self_zero_slip_hessian=c.native_zero_slip_hessian=='both',
                friction_epsilon=c.friction_velocity,particle_enable_self_contact=True,
                particle_self_contact_radius=2*c.cloth_radius,particle_self_contact_margin=4*c.cloth_radius,
                particle_topological_contact_filter_threshold=c.self_contact_filter_hops,particle_vertex_contact_buffer_size=64,
                particle_edge_contact_buffer_size=128,rigid_body_contact_buffer_size=512,
                rigid_body_particle_contact_buffer_size=32768)
        def collision(v):
            pipeline=newton.CollisionPipeline(v,broad_phase='nxn',soft_contact_margin=c.cloth_contact_margin,
                contact_matching='latest' if c.rigid_contact_history else 'disabled',
                enable_rigid_soft_full_surface_contact=True,soft_contact_max=c.contact_capacity)
            if c.native_analytic_box_contacts:
                from .box_contacts import install
                install(pipeline,v,stable=c.native_stable_box_contacts)
            if c.native_shape_grouped_contacts:
                from .contact_order import group_soft_pairs
                group_soft_pairs(pipeline)
            return pipeline
        self.coupler=SolverCoupledProxy(model=m,entries=[
            SolverCoupledProxy.Entry(name='rigid',solver=lambda v:SolverMuJoCo(v,use_mujoco_contacts=False,
                integrator='implicitfast',cone='elliptic',solver='newton',iterations=c.rigid_iterations,
                ls_iterations=20,nconmax=c.contact_capacity,njmax=c.contact_capacity*3),bodies=bodies,joints=joints),
            SolverCoupledProxy.Entry(name='cloth',solver=cloth_solver,bodies=props,joints=prop_joints,
                particles=list(range(m.particle_count)))],
            coupling=SolverCoupledProxy.Config(proxies=[SolverCoupledProxy.Proxy(source='rigid',destination='cloth',
                bodies=bodies,joints=finger_joints,mass_scale=1.,mode='lagged',collision_pipeline=collision,collide_interval=1)],iterations=c.native_coupling_iterations))
        self.native=self.coupler.solver('cloth');self.rigid=self.coupler.solver('rigid')
        material_neighborhood(self.native,sim)
        # The native coupler allocates this buffer during construction. Bind
        # it now, without detecting contacts or stepping, so tick-zero finite
        # checks and snapshots have the same buffer identity as later actions.
        self.native.active_contacts=self.coupler.get_proxy_contacts('rigid','cloth')
        if self.native.active_contacts is None:raise RuntimeError('Native proxy contact buffer was not initialized')
        # Native force-query output arrays otherwise grow on the first trace,
        # changing snapshot layout between tick zero and the first action.
        capacity=self.native.active_contacts.rigid_contact_max
        for name in ('_rigid_contact_body0','_rigid_contact_body1'):
            setattr(self.native,name,wp.full(capacity,-1,dtype=wp.int32,device=m.device))
        for name in ('_rigid_contact_point0_world','_rigid_contact_point1_world'):
            setattr(self.native,name,wp.zeros(capacity,dtype=wp.vec3,device=m.device))
        self.entry=self.coupler._entries['cloth'];self.rigid_entry=self.coupler._entries['rigid']
        if self.native.local_frame is not None:
            for field in ('state_0','state_1','state_tmp'):
                self.native.local_frame.register_state(getattr(self.entry,field))
            self.native.local_frame.register_contacts(self.native.active_contacts)
        self.collision=SimpleNamespace(forces=wp.zeros(c.contact_capacity,dtype=wp.vec3,device=m.device),
            wrenches=wp.zeros(m.body_count,dtype=wp.spatial_vector,device=m.device))
        if self.native.record.forces.shape[0]!=c.contact_capacity:
            self.native.record.forces=wp.zeros(c.contact_capacity,dtype=wp.vec3,device=m.device)
        self.shapes=self.entry.attribute_local_to_global[m.AttributeFrequency.SHAPE]
        # All physical particles belong to this entry. Do not assume identity if
        # the native view projection ever changes.
        if not np.array_equal(self.entry.particle_local_to_global.numpy(),np.arange(m.particle_count)):
            raise RuntimeError('Unexpected particle order: wet mass projection needs updating')

    def step(self,state_in,state_out,control,contacts,dt):
        wp.copy(self.entry.view.particle_mass,self.model.particle_mass)
        wp.copy(self.entry.view.particle_inv_mass,self.model.particle_inv_mass)
        self.coupler.step(state_in,state_out,control,contacts,dt)
        source=self.native.active_contacts
        wp.copy(contacts.soft_contact_count,source.soft_contact_count)
        for name in ('barycentric','body_pos','body_vel','normal'):
            wp.copy(getattr(contacts,'soft_contact_'+name),getattr(source,'soft_contact_'+name))
        wp.launch(remap_soft,dim=contacts.soft_contact_max,inputs=[source.soft_contact_count,
            source.soft_contact_shape,source.soft_contact_indices,self.shapes,self.entry.particle_local_to_global],
            outputs=[contacts.soft_contact_shape,contacts.soft_contact_indices],device=self.device)
        wp.copy(self.collision.forces,self.native.record.forces)
        self.collision.wrenches.zero_()
        wp.launch(remap_wrench,dim=self.entry.view.body_count,
            inputs=[self.native.record.wrenches,self.entry.body_local_to_global],outputs=[self.collision.wrenches],device=self.device)

    def check_capacity(self):
        for contact_set in (self.native.active_contacts,):
            for kind in ('soft','rigid'):
                number=int(getattr(contact_set,kind+'_contact_count').numpy()[0])
                limit=getattr(contact_set,kind+'_contact_max')
                if number>limit:raise OverflowError(f'Native global {kind} contact overflow: {number} > {limit}')
        for field,limit in (('body_particle_contact_counts',32768),('body_body_contact_counts',512)):
            n=int(getattr(self.native,field).numpy().max())
            if n>limit:raise OverflowError(f'Native {field} overflow: {n} > {limit}')

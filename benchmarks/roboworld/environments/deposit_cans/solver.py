"""Native coupled finite-effort robots and physical thin surfaces.
Assembled bag-lock springs have moving frame anchors and opposite reactions;
they are never available as grasp/attachment actions. The reaction is lagged by
one physics substep, explicitly subject to load/convergence qualification.
"""
import numpy as np
import newton
import warp as wp
from newton.solvers import SolverMuJoCo,SolverVBD
from newton.solvers.experimental.coupled import SolverCoupledProxy
from .bending import difference,redirect

class ShellVBD(SolverVBD):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Newton initially aliases this immutable rest reference with the
        # detector's mutable vertex-position buffer. At a later action boundary
        # the detector references a stepped state instead. Give rest geometry
        # stable independent storage so a fresh-instance restore cannot overwrite
        # it while restoring the detector's different position values.
        self.particle_q_rest=wp.clone(self.particle_q_rest)
        self._particle_method=redirect(SolverVBD._solve_particle_iteration)
        self.refinement=None
        self.global_refinement=None
        self.global_only=False;self.global_stride=16
        from .relative_step import RelativeStep
        self.relative_step=RelativeStep(self)
        self.global_start='previous';self.initializing=False
        self.refresh_self_contact=False  # Isolated numerical comparison only.
        self.adaptive_self_contact_refresh=True
        # Finalization advances native body_q_prev. Preserve the actual history
        # used during this solve for read-only post-step force evaluation.
        self.evidence_body_q_prev=wp.clone(self.body_q_prev)
    def _initialize_rigid_bodies(self,state_in,control,contacts,dt,refresh):
        super()._initialize_rigid_bodies(state_in,control,contacts,dt,refresh)
        # Snapshot AFTER first/reset-step history rebaselining, before solves.
        wp.copy(self.evidence_body_q_prev,self.body_q_prev)
    def _initialize_particles(self,state_in,state_out,dt):
        if self.global_refinement and self.global_refinement.precision:
            self.global_refinement.precision.initialize(state_in,dt)
        self.initializing=True
        try:super()._initialize_particles(state_in,state_out,dt)
        finally:self.initializing=False
    def _penetration_free_truncation(self,q):
        if self.global_only:
            if self.initializing and self.global_start=='previous':
                # Keep native forward_step's inertia/previous position exactly.
                # Only the nonlinear initial iterate changes. An unconstrained
                # predictor may acquire strain/contact energy before the first
                # residual-decreasing correction is attempted.
                self.particle_displacements.zero_()
                wp.copy(q,self.particle_q_prev)
                return
            self.relative_step.apply(self.particle_displacements,self.particle_displacements,q)
        else:
            super()._penetration_free_truncation(q)
    def _finalize_particles(self,state_out,dt):
        super()._finalize_particles(state_out,dt)
        if self.global_refinement and self.global_refinement.precision:
            from .elastic_precision import velocity
            p=self.global_refinement.precision
            wp.launch(velocity,self.model.particle_count,[p.current,p.previous,wp.float64(dt),state_out.particle_qd],device=self.device)
            wp.copy(p.emitted,state_out.particle_q)

    def _solve_particle_iteration(self,state_in,state_out,contacts,dt,iter_num):
        if self.global_only:
            if iter_num==0:self._collision_detection_penetration_free(state_in)
            if iter_num%self.global_stride==0 or iter_num==self.iterations-1:
                if self.refresh_self_contact and iter_num:
                    self._collision_detection_penetration_free(state_in)
                elif self.adaptive_self_contact_refresh and iter_num:
                    self.relative_step.refresh_if_needed(state_in)
                self.global_refinement.refine(state_in,state_out,contacts,dt)
            wp.copy(state_out.particle_q,state_in.particle_q)
            return
        if self.refinement and iter_num==0:
            for _ in range(10):self.refinement.refine(state_in,state_out,contacts,dt)
        if self.global_refinement and iter_num==0:self.global_refinement.refine(state_in,state_out,contacts,dt)
        self._particle_method(self,state_in,state_out,contacts,dt,iter_num)
        if self.refinement and iter_num%4==3:self.refinement.refine(state_in,state_out,contacts,dt)
        if self.global_refinement and (iter_num%self.global_stride==self.global_stride-1 or iter_num==self.iterations-1):self.global_refinement.refine(state_in,state_out,contacts,dt)


@wp.kernel
def follow_frame(ids:wp.array[int],bodies:wp.array[int],local:wp.array[wp.vec3],
                 bq:wp.array[wp.transform],q:wp.array[wp.vec3]):
    i=wp.tid();q[ids[i]]=wp.transform_point(bq[bodies[i]],local[i])

@wp.kernel
def retention_reaction(particles:wp.array[int],bodies:wp.array[int],local:wp.array[wp.vec3],
    stiffness:float,bq:wp.array[wp.transform],com:wp.array[wp.vec3],q:wp.array[wp.vec3],
    forces:wp.array[wp.spatial_vector],record:wp.array[wp.vec3]):
    i=wp.tid();body=bodies[i];x=wp.transform_point(bq[body],local[i])
    f=stiffness*(q[particles[i]]-x)
    r=x-wp.transform_point(bq[body],com[body]);record[i]=f
    wp.atomic_add(forces,body,wp.spatial_vector(f,wp.cross(r,f)))

@wp.kernel
def plastic_bending(q:wp.array[wp.vec3],edges:wp.array2d[int],lengths:wp.array[float],
    properties:wp.array2d[float],limit:wp.array[float],rest:wp.array[float],dissipated:wp.array[float]):
    i=wp.tid()
    if limit[i]<=0.:return
    a=edges[i,0];b=edges[i,1];c=edges[i,2];d=edges[i,3]
    if a<0 or b<0:return
    n1=wp.cross(q[c]-q[a],q[d]-q[a]);n2=wp.cross(q[d]-q[b],q[c]-q[b]);edge=q[d]-q[c]
    if wp.length(n1)<1e-6 or wp.length(n2)<1e-6 or wp.length(edge)<1e-6:return
    n1=wp.normalize(n1);n2=wp.normalize(n2);edge=wp.normalize(edge)
    theta=wp.atan2(wp.dot(wp.cross(n1,n2),edge),wp.dot(n1,n2))
    strain=difference(theta,rest[i])
    if wp.abs(strain)>limit[i]:
        residual=wp.sign(strain)*limit[i]
        rest[i]=theta-residual
        dissipated[i]+=0.5*properties[i,0]*lengths[i]*(strain*strain-residual*residual)

class Coupled:
    def __init__(self,sim,*,monolithic=False):
        self.model=m=sim.model;self.device=m.device;c=sim.config
        self.monolithic=monolithic;self.motor_controller=None
        self.pin_particles=wp.array([p[0] for p in sim.assembly.pins],dtype=int,device=m.device)
        self.pin_bodies=wp.array([p[1] for p in sim.assembly.pins],dtype=int,device=m.device)
        self.pin_local=wp.array([p[2] for p in sim.assembly.pins],dtype=wp.vec3,device=m.device)
        self.anchor_ids=wp.array(sim.assembly.anchor_ids,dtype=int,device=m.device)
        self.retention_forces=wp.zeros(len(sim.assembly.pins),dtype=wp.vec3,device=m.device)
        self.yield_angles=wp.array(sim.assembly.plastic_yield,dtype=float,device=m.device)
        self.plastic_dissipation=wp.zeros(m.edge_count,dtype=float,device=m.device)
        def surfaces(view):
            # Builder coloring covers joints, not future dynamic contacts.
            # Simultaneous tray/cup updates diverged at tick38; serial native
            # Gauss-Seidel with identical physical constants settles them.
            # Singleton groups are a conservative graph coloring for any
            # body-contact graph, including subsequently moved props.
            view.body_color_groups=[wp.array([i],dtype=int,device=view.device)
                for i in range(view.body_count)]
            # Zero-mass spring endpoints are assembly coordinates, not material.
            # Do not let the self-contact detector repel bag triangles from the
            # coincident helper vertices. All physical bag/can pairs remain live.
            exclusions={int(i):set(range(view.tri_count)) for i in sim.assembly.anchor_ids}
            solver=ShellVBD(view,iterations=c.iterations,particle_enable_self_contact=True,
                particle_external_vertex_contact_filtering_map=exclusions,
                particle_self_contact_radius=2*c.contact_radius,particle_self_contact_margin=c.contact_margin,
                particle_topological_contact_filter_threshold=2,particle_vertex_contact_buffer_size=256,
                particle_edge_contact_buffer_size=512,rigid_body_contact_buffer_size=512,
                # A body's list is drawn from the pipeline's complete soft-contact
                # set, so it must be able to retain that whole set in the worst case.
                rigid_body_particle_contact_buffer_size=c.contact_capacity,friction_epsilon=c.friction_velocity,
                rigid_avbd_contact_alpha=.95,rigid_avbd_linear_beta=100000.,rigid_contact_k_start=10000.)
            solver.global_stride=c.global_stride
            if c.rigid_compression_reference:
                from .rigid_compression_reference import install
                install(solver)
            # Warp's argument is a squared-length degeneracy threshold, despite
            # the Newton property name.1e-5 would collapse every edge<3.16mm.
            # Our smallest actual rim edges are submillimetric, not points.
            solver.trimesh_collision_detector.edge_edge_parallel_epsilon=1e-12
            if c.reciprocal_topology:
                from .topology import reciprocal_edge_filter
                solver.topology_correction=reciprocal_edge_filter(solver.trimesh_collision_detector)
            if c.bag_local_contact:
                from .topology import enable_bag_local_contact
                bag=sim.assembly.bag_spec
                solver.bag_topology_correction=enable_bag_local_contact(
                    solver.trimesh_collision_detector,np.arange(bag['first'],bag['first']+bag['count']))
            if c.shell_numerics=='global':
                from .global_iteration import GlobalIteration
                solver.global_refinement=GlobalIteration(solver,sim.assembly.surfaces)
                solver.global_refinement.cg._maxiter=c.linear_maxiter
                solver.global_refinement.merit=c.global_merit
                solver.global_refinement.force_only_evaluation=c.force_only_evaluation
                solver.global_refinement.line_trials=c.global_line_trials
                if c.elastic_precision:
                    from .elastic_precision import ElasticPrecision
                    solver.global_refinement.precision=ElasticPrecision(solver.global_refinement)
                solver.global_refinement.linear.tiled_reductions=c.tiled_search_reductions
                if c.cache_elastic_search:
                    from .sparse_search import SparseElastic
                    solver.global_refinement.linear.elastic_cache=SparseElastic(solver.global_refinement.linear)
                solver.global_only=True
                solver.global_start=c.global_start
            return solver
        def collision(view):
            return newton.CollisionPipeline(view,broad_phase='nxn',soft_contact_margin=c.rigid_soft_margin,
                enable_rigid_soft_full_surface_contact=True,soft_contact_max=c.contact_capacity)
        robot=[i for i,n in enumerate(m.body_label) if n.startswith(('left/','right/'))]
        self.coupler=None;self.rigid=None
        if robot and not monolithic:
            others=[i for i in range(m.body_count) if i not in robot]
            child=m.joint_child.numpy();joints=[i for i,b in enumerate(child) if b in robot]
            other_joints=[i for i,b in enumerate(child) if b not in robot]
            self.coupler=SolverCoupledProxy(model=m,entries=[
                SolverCoupledProxy.Entry(name='robot',solver=lambda v:SolverMuJoCo(v,use_mujoco_contacts=False,
                    integrator='implicitfast',cone='elliptic',solver='newton',iterations=c.rigid_iterations,
                    ls_iterations=20,nconmax=c.contact_capacity,njmax=c.contact_capacity*3),bodies=robot,joints=joints),
                SolverCoupledProxy.Entry(name='surfaces',solver=surfaces,bodies=others,joints=other_joints,
                    particles=list(range(m.particle_count)))],
                coupling=SolverCoupledProxy.Config(proxies=[SolverCoupledProxy.Proxy(source='robot',destination='surfaces',
                    bodies=robot,mass_scale=1.,mode='lagged',proxy_relaxation=c.proxy_relaxation,
                    collision_pipeline=collision,collide_interval=1)],iterations=c.coupling_iterations))
            self.native=self.coupler.solver('surfaces');self.rigid=self.coupler.solver('robot')
            self.contacts=self.coupler.get_proxy_contacts('robot','surfaces')
        else:
            self.native=surfaces(m);self.contacts=None
            if monolithic and robot:
                from .unified_candidate import MotorController
                self.motor_controller=MotorController(sim,self.native)
        self.plasticity=c.plasticity;self.retention_stiffness=1/c.retention_compliance

    def prepare(self,state):
        if self.motor_controller:self.motor_controller.prepare(state)
        wp.launch(retention_reaction,dim=self.pin_particles.size,inputs=[self.pin_particles,self.pin_bodies,self.pin_local,
            self.retention_stiffness,state.body_q,self.model.body_com,state.particle_q],outputs=[state.body_f,self.retention_forces],device=self.device)
        wp.launch(follow_frame,dim=self.anchor_ids.size,inputs=[self.anchor_ids,self.pin_bodies,self.pin_local,state.body_q],outputs=[state.particle_q],device=self.device)

    def step(self,state_in,state_out,control,contacts,dt):
        g=self.native.global_refinement
        if g is not None:
            # Captured ping-pong graphs use fixed slot indices. Clear/reset
            # within EACH graph so a trace never mixes two different steps.
            g.metrics.zero_()
            g.metric_slot=0
        if g and g.precision:g.precision.prepare(state_in)
        if self.coupler:
            wp.copy(self.native.model.edge_rest_angle,self.model.edge_rest_angle)
            self.coupler.step(state_in,state_out,control,contacts,dt)
        else:
            self.contacts=contacts;self.native.step(state_in,state_out,control,contacts,dt)
            if self.monolithic:
                newton.eval_ik(self.model,state_out,state_out.joint_q,state_out.joint_qd)
        if self.plasticity:
            m=self.model
            wp.launch(plastic_bending,dim=m.edge_count,inputs=[state_out.particle_q,m.edge_indices,m.edge_rest_length,
                m.edge_bending_properties,self.yield_angles],outputs=[m.edge_rest_angle,self.plastic_dissipation],device=self.device)

    def check(self):
        native=self.native
        search=getattr(native.global_refinement,'surface_search',None)
        if search is not None and int(search.overflow.numpy()[0]):
            raise OverflowError('Coupled surface-contact search capacity exceeded')
        detector=native.trimesh_collision_detector
        if np.any(detector.resize_flags.numpy()):
            maxima={kind:int(getattr(detector,kind).numpy().max()) for kind in
                ('vertex_colliding_triangles_count','edge_colliding_edges_count')}
            raise OverflowError('Surface self-contact capacity exceeded: '+str(maxima))
        if native.global_only and int(native.relative_step.degenerate.numpy()[0]):raise ValueError('Intersecting/degenerate initial self-contact pair')
        if self.contacts is not None:
            for kind in ('soft','rigid'):
                count=int(getattr(self.contacts,kind+'_contact_count').numpy()[0])
                if count>getattr(self.contacts,kind+'_contact_max'):raise OverflowError(kind+' contact capacity exceeded')
        for name,capacity in [
            ('body_particle_contact_counts',native.body_particle_contact_buffer_pre_alloc),
            ('body_body_contact_counts',native.body_body_contact_buffer_pre_alloc),
        ]:
            counts=getattr(native,name,None)
            if counts is not None and counts.size and int(counts.numpy().max())>capacity:raise OverflowError(name+' overflow')

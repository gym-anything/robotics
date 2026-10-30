"""Privileged every-substep telemetry; retention and gaps remain explicit.

This is recording, never a controller, rollback or additional physics step.
Native final contact forces are not described as real sensor measurements.
"""
from dataclasses import asdict
import shutil
import numpy as np
import warp as wp
from gym_anything_robotics.physics_trace import PhysicsTraceStore,ValidatedJSON
from .contact_evidence import ContactEvidence
from .gripper import state as gripper_state

@wp.kernel
def summarize_residual(force:wp.array[wp.vec3],groups:wp.array[int],out:wp.array2d[wp.float64]):
    i=wp.tid();g=groups[i]
    if g<0:return
    f=force[i]
    for j in range(3):wp.atomic_add(out,g,j,wp.float64(f[j]))
    wp.atomic_add(out,g,3,wp.float64(wp.dot(f,f)))

class Trace:
    def __init__(self,env,path,*,max_bytes=1024**3,columnar_contacts=True):
        # Contact-rich event streams exceed100k rows in seconds. Keep every
        # sample/event within the explicit byte budget; report any byte eviction.
        self.env=env;self.store=PhysicsTraceStore(path,max_bytes=max_bytes,
            max_items=max(100000,max_bytes//64),compression='zstd')
        self.current=None;self.previous=set();self.contacts=ContactEvidence(env)
        self.columnar_contacts=bool(columnar_contacts)
        self.c=env.solver.native if hasattr(env.solver,'graphs') else env.solver
        self.s=self.c.native
        from .self_contact_evidence import SelfContacts
        self.self_contacts=SelfContacts(self.s,2*env.config.contact_capacity)
        self.previous_self=set()
        self.source=self.destination=None
        self.actuated_dofs=np.asarray(sorted(spec['dof']for spec in env.assembly.robot_joints.values()),np.int32)
        self.residual_summary=None
        if self.s.global_refinement:
            self.residual_summary=wp.zeros((len(env.assembly.surfaces),4),dtype=wp.float64,device=env.model.device)
        self.body_map=np.arange(env.model.body_count);self.shape_map=np.arange(env.model.shape_count)
        if self.c.coupler:
            self.source=self.c.coupler._entries['robot'];self.destination=self.c.coupler._entries['surfaces']
            self.body_map=self.destination.body_local_to_global.numpy()
            self.shape_map=self.destination.attribute_local_to_global[env.model.AttributeFrequency.SHAPE].numpy()
    def begin(self,context):
        if self.current:raise RuntimeError('Action trace already active')
        if shutil.disk_usage(self.store.path.parent).free<512*1024**2:
            raise OSError('Trace disk reserve unavailable; no action executed')
        e=self.env;key=str(context['action_id'])
        self.previous={(r['surface'],r['body'],r['shape'])for r in self.contacts.read()} if e.tick else set()
        self.previous_self=self.self_contacts.identities(self.self_contacts.read(e.state.particle_q,e.config.dt)) if e.tick else set()
        self.store.begin(key,str(context['request_id']),e.tick,e.config.dt,dict(
            source='deposit-cans/newton1.5',generation=e.generation,config=asdict(e.config),
            robot_solver='partitioned_mujoco_vbd'if self.c.coupler else'unified_bounded_vbd_candidate',
            wsg_synchronizer=dict(formulation='internal ALM',metric_n_m=self.s.wsg_synchronizer.rho)
                if hasattr(self.s,'wsg_synchronizer')else None,
            contact_preconditioner=('nonsymmetric_sparse_direct'if hasattr(self.c,'residual_solver')
                else'local6x6'if hasattr(self.s.global_refinement,'contact_coarse')else'legacy'),
            force_solver_revision=e.config.solver_revision,
            self_contact_search=dict(
                force_accumulation='float64 residual'if hasattr(self.s.global_refinement,'accurate_residual')else'legacy',
                geometry='float64 closest features'if self.self_contacts.precise_geometry else'float32',
                precise_slip=getattr(getattr(self.s.global_refinement,'surface_search',None),'precise_slip',False),
                exact_fixed_normal_tangent=getattr(getattr(self.s.global_refinement,'surface_search',None),'exact_slip_tangent',False)),
            sampling='Every completed physics substep; no additional physics steps.',
            rigid_shell_contact_encoding=('rigid-shell-contact-columns-v1'if self.columnar_contacts else'legacy-row-list'),
            rigid_shell_contact_columns='New column-form contacts retain count and aligned surface/body/shape names, point_world_m and force_on_surface_n arrays, plus the common source string. deposit_cans.contact_records.rows expands either schema.',
            body_labels=e.model.body_label,shape_labels=e.model.shape_label,
            initial_contacts=sorted(self.previous),
            initial_self_contacts=sorted(self.previous_self),
            destination_body_to_global=self.body_map.tolist(),destination_shape_to_global=self.shape_map.tolist(),
            surfaces=[dict(name=a['name'],first=a['first'],count=a['count'])for a in e.assembly.surfaces],
            forces='Native VBD end-state contact evaluations; partitioned feedback or last implicit joint evaluations as labelled, NOT measured sensors or independent additive loads.',
            shell_endpoint_residual='Last actual shell refinement force plus implicit inertia, summed per surface and L2 norm; before plastic rest-angle update. Diagnostic only: no extra solve or state mutation.',
            self_contacts='kind0: directed edge/edge contribution on the first edge endpoints only; reverse candidate provides its reaction. kind1: vertex/triangle, four vertex contributions once. Sum listed vertex forces, never add an invented reverse force.',
            self_contact_capacity=2*e.config.contact_capacity,
            contact_coordinates=dict(
                self_slip='precise solver increments'if self.self_contacts.precision is not None else'public float32 positions',
                rigid_shell='precise solver positions'if self.contacts.precision is not None else'public float32 positions'),
            missing='No measured force sensors; final-iterate evaluations are not the full intra-solver force trajectory.',
            qualification='Recorder implementation pending runtime and completeness qualification.'))
        self.current=key;e.trace_callback=self.append
        return dict(available=True,id=key,store_path=str(self.store.path))
    def append(self,e):
        if shutil.disk_usage(self.store.path.parent).free<512*1024**2:
            raise OSError('Trace disk reserve reached after a completed physics step')
        rows=self.contacts.read(columnar=self.columnar_contacts)
        pairs=(set(zip(rows['surface'],rows['body'],rows['shape']))if self.columnar_contacts
            else{(r['surface'],r['body'],r['shape'])for r in rows})
        events=[dict(type=kind,tick=e.tick,surface=p[0],body=p[1],shape=p[2])for kind,delta in
            (('contact_begin',pairs-self.previous),('contact_end',self.previous-pairs))for p in sorted(delta)]
        self.previous=pairs
        shell_contacts=self.self_contacts.read(e.state.particle_q,e.config.dt)
        shell_pairs=self.self_contacts.identities(shell_contacts)
        events.extend(dict(type=kind,tick=e.tick,contact_kind=p[0],primitive0=p[1],primitive1=p[2])
            for kind,delta in (('self_contact_begin',shell_pairs-self.previous_self),
                ('self_contact_end',self.previous_self-shell_pairs)) for p in sorted(delta))
        self.previous_self=shell_pairs
        sample=dict(tick=e.tick,generation=e.generation,contact_manifold_tick=e.tick-1,
            **{k:getattr(e.state,k).numpy()for k in ('body_q','body_qd','joint_q','joint_qd','particle_q','particle_qd')},
            requested_targets=e.targets.copy(),actuator_setpoints=e.control.joint_target_q.numpy(),
            actuator_velocity_setpoints=e.control.joint_target_qd.numpy(),
            gripper_commands=gripper_state(e),
            # Keep exact numeric contact columns in the existing PT1 binary
            # array store; no float-to-text conversion or sampling reduction.
            contacts=rows if self.columnar_contacts else ValidatedJSON(rows),self_contacts=shell_contacts,edge_rest_angle=e.model.edge_rest_angle.numpy(),
            plastic_dissipation_j=self.c.plastic_dissipation.numpy(),
            retention_force_on_frame_n=self.c.retention_forces.numpy(),
            evaluation=e.evaluation)
        poses=e.state.body_q;previous=self.s.evidence_body_q_prev
        if self.c.coupler:
            sample.update(qfrc_actuator=e.rigid_solver.mjw_data.qfrc_actuator.numpy(),
                qfrc_constraint=e.rigid_solver.mjw_data.qfrc_constraint.numpy(),
                mujoco_dof_to_source_dof=e.rigid_solver.mjc_dof_to_newton_dof.numpy(),
                source_dof_to_global=self.source.joint_dof_local_to_global.numpy(),
                coupling_feedback_n_nm=self.c.coupler._proxy_mappings[0].coupling_forces.numpy(),
                proxy_body_q=self.destination.state_1.body_q.numpy(),proxy_body_qd=self.destination.state_1.body_qd.numpy())
            poses=self.destination.state_1.body_q
        elif self.c.motor_controller:
            sample['joint_scalar_evaluations']=dict(
                global_dof_indices=self.actuated_dofs,
                motor_n_or_nm=self.s.motor_evaluation.numpy()[self.actuated_dofs],
                passive_n_or_nm=self.s.passive_evaluation.numpy()[self.actuated_dofs],
                mechanical_limit_n_or_nm=self.s.limit_evaluation.numpy()[self.actuated_dofs],
                commanded_gravity_feedforward_n_or_nm=self.s.motor_feedforward.numpy()[self.actuated_dofs],
                motor_effort_bounds_n_or_nm=e.model.joint_effort_limit.numpy()[self.actuated_dofs],
                indexing='Listed global Newton actuated joint DOF indices; N for prismatic, Nm for revolute.',
                scope='Last scalar evaluation within the implicit rigid sweep, not a constant force across the substep. Passive and mechanical-stop reactions are separate from bounded motor effort; structural constraint reactions are not these scalars.')
        contacts=self.c.contacts
        b0,b1,p0,p1,force,count=self.s.collect_rigid_contact_forces(
            poses,previous,contacts,e.config.dt)
        n=int(count.numpy()[0])
        if n>contacts.rigid_contact_max:raise OverflowError('Rigid trace contact capacity exceeded')
        def mapped(array,mapping):
            ids=array.numpy()[:n];valid=ids>=0;ids[valid]=mapping[ids[valid]];return ids
        sample['rigid_contacts']=dict(body0=mapped(b0,self.body_map),body1=mapped(b1,self.body_map),
            shape0=mapped(contacts.rigid_contact_shape0,self.shape_map),
            shape1=mapped(contacts.rigid_contact_shape1,self.shape_map),
            point0_world_m=p0.numpy()[:n],point1_world_m=p1.numpy()[:n],force_world_on_body1_n=force.numpy()[:n])
        if self.s.global_refinement:
            g=self.s.global_refinement
            self.residual_summary.zero_()
            wp.launch(summarize_residual,e.model.particle_count,[g.force,g.coarse.groups,self.residual_summary],device=e.model.device)
            residual=self.residual_summary.numpy()
            sample['shell_endpoint_residual']=dict(surface_names=[a['name']for a in e.assembly.surfaces],
                net_n=residual[:,:3],l2_n=np.sqrt(residual[:,3]),
                scope='Actual last refinement buffer, F_contact+F_internal+F_inertia; no recomputation after solver or plasticity')
            if hasattr(self.c,'residual_solver'):
                sample['shell_nonlinear_convergence']=ValidatedJSON(self.c.residual_solver.status())
            elif hasattr(self.c,'bounded_solver'):
                sample['shell_nonlinear_convergence']=ValidatedJSON(self.c.bounded_solver.status())
                sample['shell_search_metrics']=g.metrics.numpy()
            else:
                sample['shell_search_metrics']=g.metrics.numpy()
                sample['shell_search_metric_scope']=dict(tick=e.tick,refinement_count=g.metric_slot,
                    semantics='Only this completed substep; unused slots zeroed before its first refinement.')
        if hasattr(self.s,'wsg_synchronizer'):
            belt=self.s.wsg_synchronizer
            sample['wsg_belt']=dict(formulation='alm',joint_pairs=belt.pairs.numpy(),
                last_multiplier_n=belt.lam.numpy(),last_residual_m=belt.error.numpy(),
                numerical_metric_n_m=belt.rho,scope='Internal mechanism constraint, not measured pressure')
        self.store.append(self.current,e.tick,sample,events)
    def finish(self,*,error=None):
        if self.current is None:return None
        key,self.current=self.current,None;self.env.trace_callback=None
        fault=error or self.env.fault
        self.store.finish(key,self.env.tick,status='failed'if fault else 'completed',error=fault)
        return self.store.summary(key)
    def close(self):
        self.finish();self.store.close()

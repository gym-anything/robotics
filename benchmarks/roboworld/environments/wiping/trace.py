"""Bounded privileged substep recording; it observes, never advances physics."""
from dataclasses import asdict
from contextlib import nullcontext
import shutil
import numpy as np
import newton
from gym_anything_robotics.physics_trace import PhysicsTraceStore


class Trace:
    def __init__(self,env,path):
        self.env=env;self.store=PhysicsTraceStore(path,max_bytes=512*1024**2,max_items=100000,compression='zstd')
        self.current=None;self.previous=set()
        solver=env.rigid_solver
        self.robot_contacts=newton.Contacts(solver.get_max_contact_count(),0,
            requested_attributes={'force'},device=env.model.device)
        self.native=env.solver.native if hasattr(env.solver,'graphs') else env.solver
        self.coupled=env.config.coupling=='native'
        self.mujoco_dof_map=solver.mjc_dof_to_newton_dof.numpy()
        if self.coupled:
            entry=self.native.rigid_entry
            self.mujoco_shape_map=entry.attribute_local_to_global[env.model.AttributeFrequency.SHAPE].numpy()
            local_to_global=entry.joint_dof_local_to_global.numpy()
            positive=self.mujoco_dof_map>=0
            self.mujoco_dof_map[positive]=local_to_global[self.mujoco_dof_map[positive]]
            self.vbd_shape_map=self.native.shapes.numpy()
            self.vbd_body_map=self.native.entry.body_local_to_global.numpy()

    def begin(self,context):
        if self.current:raise RuntimeError('Trace already active')
        if shutil.disk_usage(self.store.path.parent).free<512*1024**2:
            raise OSError('Trace disk reserve unavailable; action not executed')
        e=self.env;key=str(context['action_id'])
        n=int(e.contacts.soft_contact_count.numpy()[0])
        loaded=np.linalg.norm(e.solver.collision.forces.numpy()[:n],axis=1)>1e-8
        self.previous=set(map(int,e.contacts.soft_contact_shape.numpy()[:n][loaded]))
        coupled_forces=('Unified constrained MuJoCo rigid ownership; VBD solves cloth against finite-inertia '
                        'rigid proxies, and harvests only cloth reactions.' if
                        e.config.native_rigid_ownership=='mujoco' else
                        'Native final VBD soft/rigid contact laws; loose props and cloth solved together, '
                        'with finite-inertia robot proxy exchange.')
        self.store.begin(key,str(context['request_id']),e.tick,e.config.dt,dict(
            source='wipe-work-surfaces/newton1.5',generation=e.generation,config=asdict(e.config),
            initial_loaded_cloth_shapes=sorted(self.previous),
            body_labels=e.builder.body_label,shape_labels=e.builder.shape_label,
            sampling='Every completed physics substep. Film advances at configured 60 Hz; contact exchange every substep.',
            forces=(coupled_forces+' MuJoCo and VBD constraint records are separately labelled, not summed as independent contacts. '
                    'All body, shape and DOF maps use the global model; no measured-impulse claim.' if self.coupled else
                    'Native final VBD soft contact law on cloth; opposite body wrench applied next substep. Native MuJoCo contact export and generalized actuator forces; no measured-impulse claim.'),
            reservoirs='Sparse surface cells are exact stored kg, cloth masses and ballistic packet state recorded every substep.',
            surface_offsets=e.transport.offsets.tolist(),surfaces=[dict(name=s.name,nx=int(s.nx),ny=int(s.ny),
                cell_area_m2=s.cell_area,shape_id=s.shape,body_id=s.body,
                origin_body_m=s.origin.tolist(),u_body=s.u.tolist(),v_body=s.v.tolist(),
                size_m=list(map(float,s.size)),triangle_uv_m=None if s.triangle_uv is None else s.triangle_uv.tolist())
                for s in e.transport.surfaces]))
        self.current=key;e.trace_callback=self.append
        return dict(available=True,id=key,store_path=str(self.store.path))

    def append(self,e):
        if shutil.disk_usage(self.store.path.parent).free<512*1024**2:raise OSError('Trace disk reserve reached after physics')
        c=e.contacts;n=int(c.soft_contact_count.numpy()[0])
        soft={name:getattr(c,'soft_contact_'+name).numpy()[:n] for name in
              ('shape','indices','barycentric','body_pos','normal')}
        soft['force_world_on_cloth_n']=e.solver.collision.forces.numpy()[:n]
        loaded=np.linalg.norm(soft['force_world_on_cloth_n'],axis=1)>1e-8
        pairs=set(map(int,soft['shape'][loaded]))
        events=[dict(type=kind,tick=e.tick,shape_id=int(p)) for kind,delta in
            (('cloth_contact_begin',pairs-self.previous),('cloth_contact_end',self.previous-pairs)) for p in sorted(delta)]
        self.previous=pairs
        rc=self.robot_contacts;e.rigid_solver.update_contacts(rc);nr=int(rc.rigid_contact_count.numpy()[0])
        if nr>rc.rigid_contact_max:raise OverflowError('Trace contact capacity exceeded')
        rigid={name:getattr(rc,'rigid_contact_'+name).numpy()[:nr] for name in ('shape0','shape1','point0','point1')}
        rigid['wrench_world_on_shape0']=rc.force.numpy()[:nr]
        if self.coupled:
            for name in ('shape0','shape1'):
                ids=rigid[name];valid=ids>=0;ids[valid]=self.mujoco_shape_map[ids[valid]]
        native=e.rigid_solver.mjw_data;t=e.transport;active=np.flatnonzero(t.surface.sum(1)>0)
        sample=dict(tick=e.tick,generation=e.generation,contact_manifold_tick=e.tick-1,
            **{k:getattr(e.state,k).numpy() for k in ('body_q','body_qd','joint_q','joint_qd','particle_q','particle_qd')},
            requested_targets=e.targets,actuator_setpoints=e.control.joint_target_q.numpy(),
            qfrc_actuator=native.qfrc_actuator.numpy(),qfrc_constraint=native.qfrc_constraint.numpy(),
            mujoco_dof_to_newton_dof=self.mujoco_dof_map,
            soft_contacts=soft,rigid_contacts=rigid,cloth_kg=t.cloth.copy(),
            surface_cell_ids=active,surface_kg=t.surface[active],ledger_residual_kg=t.last_residual.copy(),
            exchange=t.step_metrics,soft_contact_reaction_wrench=e.solver.collision.wrenches.numpy())
        if self.coupled:
            native=self.native.native;contacts=native.active_contacts
            frame=getattr(native,'local_frame',None)
            # Query the same precise solve geometry. A world-float query would
            # reintroduce the rounding defect in teacher telemetry even though
            # the actual force feedback was computed in the local frame.
            with frame.scope() if frame is not None else nullcontext():
                b0,b1,p0,p1,force,count=native.collect_rigid_contact_forces(
                    self.native.entry.state_1.body_q,native.previous_for_record,contacts,e.config.dt)
            nr=int(count.numpy()[0])
            if nr>contacts.rigid_contact_max:raise OverflowError('VBD trace contact capacity exceeded')
            def mapped(array,mapping):
                ids=array.numpy()[:nr];valid=ids>=0;ids[valid]=mapping[ids[valid]];return ids
            origin=np.asarray(tuple(frame.origin),dtype=float) if frame is not None else np.zeros(3)
            sample['vbd_rigid_contacts']=dict(
                body0=mapped(b0,self.vbd_body_map),body1=mapped(b1,self.vbd_body_map),
                shape0=mapped(contacts.rigid_contact_shape0,self.vbd_shape_map),
                shape1=mapped(contacts.rigid_contact_shape1,self.vbd_shape_map),
                point0_world_m=p0.numpy()[:nr]+origin,point1_world_m=p1.numpy()[:nr]+origin,
                force_world_on_body1_n=force.numpy()[:nr],
                evaluation_frame_origin_world_m=origin)
        if t.flow:
            f=t.flow;sample['runoff']=dict(positions_m=f.positions,velocities_m_s=f.velocities,mass_kg=f.mass,
                unmapped_shape_kg=f.unmapped,escaped_kg=f.escaped)
        self.store.append(self.current,e.tick,sample,events)

    def finish(self):
        if self.current is None:return None
        key,self.current=self.current,None;self.env.trace_callback=None
        self.store.finish(key,self.env.tick,status='failed' if self.env.fault else 'completed',error=self.env.fault)
        return self.store.summary(key)

    def close(self):self.finish();self.store.close()

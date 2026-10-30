"""Explicit data-only physics checkpoints, including both solvers' caches.

Restoration is a declared discontinuity. It is never used inside a demonstration.
Numeric caches are stored as exact bytes because unused native slots can contain
nonfinite sentinels. Public physics fields are independently required to be finite.
No executable object, device pointer, kernel or pickle is serialized.
"""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
from whiteboard.gpu_runtime import _arrays as mujoco_arrays


def identity(e):
    digest=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(p.name.encode());digest.update(p.read_bytes())
    return dict(schema='wiping.checkpoint.v1',source_sha256=digest.hexdigest(),config=asdict(e.config),
        robot_sources=e.assembly.sources,body_labels=list(e.model.body_label),
        particles=e.model.particle_count,shapes=e.model.shape_count)


def arrays(e):
    out={}
    def direct(obj,prefix):
        for k,v in vars(obj).items():
            if isinstance(v,wp.array):
                a=v.numpy()
                if a.dtype.kind in 'biuf':out[prefix+k]=v
    for name in ('state_0','state_1','control','contacts'):direct(getattr(e,name),name+'/')
    native=e.solver.native if hasattr(e.solver,'graphs') else e.solver
    direct(native.collision,'reaction/');direct(native.native,'vbd/')
    frame=getattr(native.native,'local_frame',None)
    if frame is not None:
        for key,value in frame.snapshot_arrays().items():out['numerical_frame/'+key]=value
    detector=native.native.trimesh_collision_detector
    direct(detector,'self_contact/')
    # Collision-info structs contain pointers; preserve the numeric arrays they
    # reference, never serialize/reassign those pointer-bearing structs.
    if hasattr(detector,'collision_info'):direct(detector.collision_info,'self_contact_info/')
    if e.config.coupling=='native':
        direct(native.native.record,'vbd_record/')
        coupler=native.coupler
        direct(coupler,'coupler/')
        for name,entry in coupler._entries.items():
            for field in ('state_0','state_1','state_tmp','control'):
                value=getattr(entry,field)
                if value is not None:direct(value,f'entry/{name}/{field}/')
            for field in ('particle_mass','particle_inv_mass'):
                value=getattr(entry.view,field,None)
                if value is not None:out[f'entry/{name}/view/{field}']=value
        for group in ('_proxy_mappings','_proxy_particle_mappings','_proxy_joint_mappings'):
            for i,mapping in enumerate(getattr(coupler,group)):direct(mapping,f'proxy/{group}/{i}/')
        for (src,dst),collision in coupler._proxy_collision_configs.items():
            if collision.contacts is not None:direct(collision.contacts,f'proxy_contacts/{src}/{dst}/')
    for k,v in mujoco_arrays(e.rigid_solver.mjw_data):out['mujoco/'+k]=v
    for name in ('particle_mass','particle_inv_mass'):out['model/'+name]=getattr(e.model,name)
    return out


def capture(e):
    e.check_finite()
    if e.fault or e.trace_callback:raise ValueError('Checkpoint requires a healthy action boundary')
    transport={k:v.copy() for k,v in vars(e.transport).items() if isinstance(v,np.ndarray)}
    flow={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in vars(e.transport.flow).items()
          if isinstance(v,(np.ndarray,int,float))} if e.transport.flow else None
    values={}
    for k,v in arrays(e).items():
        a=np.ascontiguousarray(v.numpy());values[k]=dict(dtype=a.dtype.str,shape=list(a.shape),bytes=a.view(np.uint8).ravel().copy())
    native=e.solver.native if hasattr(e.solver,'graphs') else e.solver
    counters=[]
    if e.config.coupling=='native':
        counters=[dict(source=k[0],destination=k[1],value=v.collide_counter) for k,v in native.coupler._proxy_collision_configs.items()]
    return dict(identity=identity(e),tick=e.tick,generation=e.generation,targets=e.targets.copy(),
        water=e.water.copy(),rigid_step=e.rigid_solver._step,
        proxy_collision_counters=counters,
        numeric=values,transport=transport,flow=flow,
        exchange=copy.deepcopy(e.transport.step_metrics),contact_work_j=e.transport.contact_work_j,
        # Geometric masks are deterministic derived data, not episode state.
        progress=copy.deepcopy({k:v for k,v in vars(e.progress).items() if k not in ('target_masks','geometry')}),evaluation=copy.deepcopy(e.evaluation))


def restore(e,data):
    if e.trace_callback:raise ValueError('Finish current action trace before restoring')
    if json.dumps(data['identity'],sort_keys=True)!=json.dumps(identity(e),sort_keys=True):
        raise ValueError('Checkpoint source/configuration/topology differs; no state changed')
    target=arrays(e)
    if set(target)!=set(data['numeric']):raise ValueError('Native cache layout differs; no state changed')
    writes=[]
    for k,v in target.items():
        current=v.numpy();record=data['numeric'][k]
        if record['dtype']!=current.dtype.str or tuple(record['shape'])!=current.shape:
            raise ValueError('Checkpoint array layout differs: '+k)
        value=np.asarray(record['bytes'],np.uint8).view(current.dtype).reshape(current.shape)
        if k.startswith(('state_0/','state_1/')) and k.split('/')[1] in ('body_q','body_qd','particle_q','particle_qd','joint_q','joint_qd'):
            if not np.isfinite(value).all():raise ValueError('Nonfinite checkpoint physics')
        writes.append((v,value))
    if type(data['tick']) is not int or data['tick']<0:raise ValueError('Invalid checkpoint tick')
    for key,values in data['transport'].items():
        if key not in vars(e.transport) or not isinstance(getattr(e.transport,key),np.ndarray):raise ValueError('Unknown material field')
        if not np.isfinite(values).all():raise ValueError('Nonfinite material field')
    for k in ('targets','water'):
        if data[k].shape!=getattr(e,k).shape or not np.isfinite(data[k]).all():raise ValueError('Invalid '+k)
    for target,value in writes:target.assign(value)
    e.targets=data['targets'].copy();e.water=data['water'].copy()
    for k,v in data['transport'].items():
        current=getattr(e.transport,k)
        # Storage decoding creates C-order arrays. Preserve the live field's
        # layout (surface is Fortran-order), including its reduction order.
        if current.shape==v.shape:current[...] = v
        else:setattr(e.transport,k,v.copy(order='K'))
    if data['flow'] is not None:
        for k,v in data['flow'].items():setattr(e.transport.flow,k,v.copy() if isinstance(v,np.ndarray) else v)
    e.transport.step_metrics=copy.deepcopy(data['exchange']);e.transport.contact_work_j=data['contact_work_j']
    for k,v in data['progress'].items():setattr(e.progress,k,copy.deepcopy(v))
    e.evaluation=copy.deepcopy(data['evaluation']);e.tick=data['tick'];e.generation+=1;e.fault=None
    native=e.rigid_solver.native if hasattr(e.rigid_solver,'graphs') else e.rigid_solver
    native._step=data['rigid_step']
    if e.config.coupling=='native':
        coupled=e.solver.native if hasattr(e.solver,'graphs') else e.solver
        for record in data['proxy_collision_counters']:
            coupled.coupler._proxy_collision_configs[(record['source'],record['destination'])].collide_counter=record['value']
    # Spatial hierarchies are derived caches; rebuilding them is not a step.
    e.model.bvh_refit_shapes(e.state_0)
    return e.get_state()

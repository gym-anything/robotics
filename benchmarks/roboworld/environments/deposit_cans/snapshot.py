"""Explicit data-only checkpoint, not a continuity-preserving reset.

Exact bytes preserve native cache sentinels. Physics values must be finite.
No executable object, pointer, kernel, or pickle is accepted from storage.
Future GPU trajectories are not promised to be bitwise deterministic.
"""
from gym_anything_robotics.paths import package_dir
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
from whiteboard.gpu_runtime import _arrays as mujoco_arrays

PROGRESS=('stable_since','last_tick','floor_losses','last_inside','last_points',
    'last_mouth_z','wall_transition_suspects','ejected_preloads',
    'surface_path_failures','surface_path_checked_through','rest_history')

def _source_digest():
    digest=hashlib.sha256()
    here,robots=Path(__file__).resolve().parent,package_dir('robots')
    paths=[('deposit_cans/'+p.name,p) for p in here.glob('*.py')]
    paths += [('robots/'+p.relative_to(robots).as_posix(),p) for p in robots.rglob('*') if p.suffix in ('.py','.json') and not any(x in p.relative_to(robots).parts for x in ('tests','tools','yam'))]
    for name,p in sorted(paths):
        digest.update(name.encode());digest.update(p.read_bytes())
    return digest.hexdigest()


# Imported by env before constructing the first model. A long-lived process
# must not stamp its old in-memory solver with subsequently edited source files.
SOURCE_SHA256=_source_digest()


def identity(e):
    return dict(schema='deposit_cans.checkpoint.v2',source_sha256=SOURCE_SHA256,
        numerical_revision=e.config.solver_revision,
        force_solver=('residual-newton-v5'if hasattr(native(e),'residual_solver')else
            'bounded-shell-v6'if hasattr(native(e),'bounded_solver')else'legacy'),
        implicit_retention=hasattr(native(e).native,'_implicit_retention'),
        contact_reduction_lanes=getattr(native(e).native,'contact_reduction_lanes',4),
        solver_mode='partitioned_mujoco_vbd'if native(e).coupler else'unified_bounded_vbd_candidate',
        dependency_schedule=hasattr(native(e).native,'_rigid_schedule'),
        dependency_schedule_mode=getattr(getattr(native(e).native,'_rigid_schedule',None),'mode','none'),
        wsg_synchronizer=hasattr(native(e).native,'wsg_synchronizer'),
        wsg_numerical_metric_n_m=getattr(getattr(native(e).native,'wsg_synchronizer',None),'rho',None),
        contact_coarse=hasattr(native(e).native.global_refinement,'contact_coarse'),
        precise_self_contact_search=getattr(getattr(native(e).native.global_refinement,'surface_search',None),'precise_slip',False),
        exact_self_contact_tangent=getattr(getattr(native(e).native.global_refinement,'surface_search',None),'exact_slip_tangent',False),
        config=asdict(e.config),robot_sources=e.assembly.sources,robot_setup=e.assembly.robot_setup.to_dict(),
        robot_runtime=e.robot_rig.capabilities(),
        body_labels=list(e.model.body_label),particles=e.model.particle_count,
        shapes=e.model.shape_count)

def native(e):return e.solver.native if hasattr(e.solver,'graphs') else e.solver

def arrays(e):
    result={}
    def direct(obj,prefix):
        if obj is None:return
        for k,v in vars(obj).items():
            # Transient query alias is always selected afresh by evaluate;
            # its owning precise arrays are recorded separately below.
            if k in ('precise_query','contact_query'):continue
            if isinstance(v,wp.array) and v.numpy().dtype.kind in 'biuf':result[prefix+k]=v
    for name in ('state_0','state_1','control','contacts'):direct(getattr(e,name),name+'/')
    c=native(e);s=c.native;direct(c,'coupled/');direct(s,'vbd/')
    direct(s.trimesh_collision_detector,'self_contact/')
    direct(getattr(s.trimesh_collision_detector,'collision_info',None),'self_contact_info/')
    direct(s.relative_step,'relative_step/')
    g=s.global_refinement
    if g:
        direct(g.precision,'global/precision/')
        direct(getattr(g,'accurate_residual',None),'global/accurate_residual/')
        direct(getattr(g,'contact_coarse',None),'global/contact_coarse/')
        direct(g,'global/');direct(g.contact,'global/contact/');direct(g.coarse,'global/coarse/')
        direct(g.linear,'global/linear/');direct(g.cg,'global/cg/');direct(g.work,'global/work/')
        direct(getattr(g.linear,'elastic_cache',None),'global/elastic_cache/')
        direct(getattr(g,'surface_search',None),'global/surface_search/')
    coupler=c.coupler;direct(coupler,'coupler/')
    direct(getattr(s,'_rigid_schedule',None),'rigid_schedule/')
    direct(getattr(s,'wsg_synchronizer',None),'wsg_synchronizer/')
    for name,entry in getattr(coupler,'_entries',{}).items():
        for field in ('state_0','state_1','state_tmp','control'):
            direct(getattr(entry,field,None),f'entry/{name}/{field}/')
        result[f'entry/{name}/edge_rest_angle']=entry.view.edge_rest_angle
    for group in ('_proxy_mappings','_proxy_particle_mappings','_proxy_joint_mappings'):
        for i,mapping in enumerate(getattr(coupler,group,[])):direct(mapping,f'proxy/{group}/{i}/')
    for (src,dst),collision in getattr(coupler,'_proxy_collision_configs',{}).items():
        direct(collision.contacts,f'proxy_contacts/{src}/{dst}/')
    if e.rigid_solver is not None:
        for k,v in mujoco_arrays(e.rigid_solver.mjw_data):result['mujoco/'+k]=v
    result['model/edge_rest_angle']=e.model.edge_rest_angle
    from .gripper import DRIVE_ARRAYS
    for name in DRIVE_ARRAYS:result['model/'+name]=getattr(e.model,name)
    return result

def capture(e):
    with e.lock:
        e.check_finite()
        if e.fault or e.trace_callback:raise ValueError('Checkpoint requires a healthy action boundary')
        values={}
        for k,v in arrays(e).items():
            a=np.ascontiguousarray(v.numpy());values[k]=dict(dtype=a.dtype.str,shape=list(a.shape),bytes=a.view(np.uint8).ravel().copy())
        progress={}
        for key in PROGRESS:
            value=getattr(e.progress,key)
            progress[key]=sorted(value) if isinstance(value,set) else copy.deepcopy(value)
        c=native(e)
        residual=getattr(c,'residual_solver',None)
        return dict(identity=identity(e),tick=e.tick,generation=e.generation,
            residual_solver_state=residual.checkpoint_state()if residual is not None else None,
            targets=e.targets.copy(),rigid_step=e.rigid_solver._step if e.rigid_solver is not None else None,numeric=values,
            collision_counters=[dict(source=k[0],destination=k[1],value=v.collide_counter)
                for k,v in getattr(c.coupler,'_proxy_collision_configs',{}).items()],
            entry_output_valid=c.coupler._entry_output_state_valid if c.coupler else None,
            progress=progress,evaluation=copy.deepcopy(e.evaluation))

def restore(e,data):
    with e.lock:
        if e.trace_callback:raise ValueError('Finish the active trace before restoring')
        if json.dumps(data['identity'],sort_keys=True)!=json.dumps(identity(e),sort_keys=True):
            raise ValueError('Checkpoint source/configuration/topology differs; no state changed')
        target=arrays(e)
        if set(target)!=set(data['numeric']):raise ValueError('Native cache layout differs; no state changed')
        writes=[]
        for k,v in target.items():
            current=v.numpy();record=data['numeric'][k]
            if record['dtype']!=current.dtype.str or tuple(record['shape'])!=current.shape:
                raise ValueError('Checkpoint array layout differs: '+k)
            blob=np.asarray(record['bytes'],np.uint8)
            if blob.nbytes!=current.nbytes:raise ValueError('Checkpoint byte count differs: '+k)
            value=blob.view(current.dtype).reshape(current.shape)
            if k.startswith(('state_0/','state_1/')) and k.split('/')[1] in ('body_q','body_qd','particle_q','particle_qd','joint_q','joint_qd'):
                if not np.isfinite(value).all():raise ValueError('Nonfinite checkpoint physics')
            writes.append((v,value))
        if type(data['tick']) is not int or data['tick']<0:raise ValueError('Invalid checkpoint tick')
        if data['targets'].shape!=e.targets.shape or not np.isfinite(data['targets']).all():raise ValueError('Invalid actuator targets')
        if set(data['progress'])!=set(PROGRESS):raise ValueError('Invalid progress fields')
        c=native(e);counter_writes=[]
        clocks=getattr(c.coupler,'_proxy_collision_configs',{})
        for record in data['collision_counters']:
            key=(record['source'],record['destination']);value=record['value']
            if key not in clocks or type(value)is not int or value<0:
                raise ValueError('Invalid collision clock')
            counter_writes.append((clocks[key],value))
        if len(counter_writes)!=len(clocks):raise ValueError('Missing collision clocks')
        if e.rigid_solver is None and data['rigid_step'] is not None:raise ValueError('Unexpected MuJoCo counter')
        if not c.coupler and data['entry_output_valid'] is not None:raise ValueError('Unexpected proxy output state')
        residual=getattr(c,'residual_solver',None)
        residual_state=data.get('residual_solver_state')
        if residual is not None:
            residual.validate_checkpoint(residual_state)
            if bool(residual_state)!=(data['tick']>0):raise ValueError('Residual checkpoint clock/status mismatch')
        elif residual_state is not None:raise ValueError('Unexpected residual solver state')
        # All data/layout validation precedes the first device write.
        for target,value in writes:target.assign(value)
        if residual is not None:residual.restore_checkpoint(residual_state)
        e.targets=data['targets'].copy();e.tick=data['tick'];e.generation+=1;e.fault=None
        if e.rigid_solver is not None:e.rigid_solver._step=data['rigid_step']
        for collision,value in counter_writes:collision.collide_counter=value
        if c.coupler:c.coupler._entry_output_state_valid=bool(data['entry_output_valid'])
        for key in PROGRESS:
            value=copy.deepcopy(data['progress'][key])
            if isinstance(getattr(e.progress,key),set):value=set(value)
            setattr(e.progress,key,value)
        e.evaluation=copy.deepcopy(data['evaluation'])
        e.progress.bag.update(e.state.particle_q.numpy())
        e.progress.integrity.previous=np.asarray(e.state.particle_q.numpy()[e.progress.integrity.ids],np.float64).copy()
        e.model.bvh_refit_shapes(e.state)
        return e.get_state()

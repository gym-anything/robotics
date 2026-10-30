"""Cache the existing elastic SEARCH operator once per nonlinear iterate.

This is not a new constitutive model or solve. It expands the very same
linear64.membrane/cached_hinges products into 3x3 BSR blocks. It is disabled
until product/translation-null and measured full-step comparisons pass.
Topology is built once; geometry-dependent values are refreshed per solve.
Warp BSR documentation and the installed bsr_mv implementation were inspected.
"""
import numpy as np
import warp as wp
from warp.sparse import bsr_zeros,bsr_set_from_triplets,bsr_mv
from .relative_step import double

@wp.kernel
def membrane_blocks(q:wp.array[wp.vec3],tri:wp.array2d[int],pose:wp.array[wp.mat22],
    area:wp.array[float],mat:wp.array2d[float],dt:wp.float64,
    indices:wp.array[int],out:wp.array[wp.mat33d]):
    t=wp.tid();a=tri[t,0];b=tri[t,1];c=tri[t,2];P=pose[t]
    p00=wp.float64(P[0,0]);p01=wp.float64(P[0,1]);p10=wp.float64(P[1,0]);p11=wp.float64(P[1,1])
    f0=(double(q[b])-double(q[a]))*p00+(double(q[c])-double(q[a]))*p10
    f1=(double(q[b])-double(q[a]))*p01+(double(q[c])-double(q[a]))*p11
    u=wp.vec3d(-p00-p10,p00,p10);v=wp.vec3d(-p01-p11,p01,p11)
    shear=wp.float64(mat[t,0])+wp.float64(2.*mat[t,2])/dt
    lam=wp.float64(mat[t,1]);A=wp.float64(area[t])
    for i in range(3):
        for j in range(3):
            H=A*(wp.float64(2.)*shear*(u[i]*u[j]*wp.outer(f0,f0)+v[i]*v[j]*wp.outer(f1,f1))
                 +shear*wp.outer(u[i]*f1+v[i]*f0,v[j]*f0+u[j]*f1)
                 +lam*wp.outer(u[i]*f0+v[i]*f1,u[j]*f0+v[j]*f1))
            wp.atomic_add(out,indices[9*t+3*i+j],H)

@wp.kernel
def hinge_blocks(grad:wp.array[wp.vec3d],edges:wp.array2d[int],length:wp.array[float],
    stiffness:wp.array2d[float],dt:wp.float64,indices:wp.array[int],out:wp.array[wp.mat33d]):
    e=wp.tid()
    if edges[e,0]<0 or edges[e,1]<0:return
    k=wp.float64(length[e])*(wp.float64(stiffness[e,0])+wp.float64(stiffness[e,1])/dt)
    for i in range(4):
        for j in range(4):
            wp.atomic_add(out,indices[16*e+4*i+j],k*wp.outer(grad[4*e+i],grad[4*e+j]))

class SparseElastic:
    def __init__(self,linear):
        self.linear=linear;g=linear.g;m=g.s.model;d=m.device;n=m.particle_count
        tri=m.tri_indices.numpy().astype(np.int64);edges=m.edge_indices.numpy().astype(np.int64)
        tk=(tri[:,:,None]*n+tri[:,None,:]).reshape(-1)
        ek=(edges[:,:,None]*n+edges[:,None,:]).reshape(-1)
        valid=np.repeat((edges[:,0]>=0)&(edges[:,1]>=0),16)
        unique=np.unique(np.r_[tk,ek[valid]])
        rows=wp.array((unique//n).astype(np.int32),dtype=int,device=d)
        cols=wp.array((unique%n).astype(np.int32),dtype=int,device=d)
        self.A=bsr_zeros(n,n,wp.mat33d,device=d)
        bsr_set_from_triplets(self.A,rows,cols,prune_numerical_zeros=False)
        nnz=self.A.nnz_sync();actual_rows=np.repeat(np.arange(n),np.diff(self.A.offsets.numpy()))
        if nnz!=len(unique) or not np.array_equal(actual_rows*n+self.A.columns.numpy()[:nnz],unique):
            raise RuntimeError('BSR topology ordering mismatch')
        self.triangle_map=wp.array(np.searchsorted(unique,tk).astype(np.int32),dtype=int,device=d)
        emap=np.zeros(len(ek),np.int32);emap[valid]=np.searchsorted(unique,ek[valid])
        self.edge_map=wp.array(emap,dtype=int,device=d)
        self.nnz=nnz
    def prepare(self):
        g=self.linear.g;m=g.s.model;d=m.device;self.A.values.zero_()
        wp.launch(membrane_blocks,m.tri_count,[g.q,m.tri_indices,m.tri_poses,m.tri_areas,m.tri_materials,
            wp.float64(g.dt),self.triangle_map,self.A.values],device=d)
        wp.launch(hinge_blocks,m.edge_count,[self.linear.hinge_grad,m.edge_indices,m.edge_rest_length,
            m.edge_bending_properties,wp.float64(g.dt),self.edge_map,self.A.values],device=d)
    def add_product(self,x,out):
        bsr_mv(self.A,x,out,alpha=1.,beta=1.)

def audit(sim,g,path):
    import json,time
    from .linear64 import membrane,cached_hinges
    m=g.s.model;d=m.device;linear=g.linear;linear.prepare_geometry()
    candidate=SparseElastic(linear);candidate.prepare();wp.synchronize_device(d)
    rng=np.random.default_rng(181)
    inputs=[rng.normal(size=(m.particle_count,3)),np.tile([.001,-.0005,.002],(m.particle_count,1))]
    errors=[];x=wp.zeros(m.particle_count,dtype=wp.vec3d,device=d)
    old=wp.zeros_like(x);new=wp.zeros_like(x)
    def reference():
        old.zero_()
        wp.launch(membrane,m.tri_count,[g.q,x,m.tri_indices,m.tri_poses,m.tri_areas,m.tri_materials,wp.float64(g.dt),old],device=d)
        wp.launch(cached_hinges,m.edge_count,[linear.hinge_grad,x,m.edge_indices,m.edge_rest_length,m.edge_bending_properties,wp.float64(g.dt),old],device=d)
    def cached():new.zero_();candidate.add_product(x,new)
    for name,value in zip(('random','uniform_translation'),inputs):
        x.assign(value);reference();cached();a=old.numpy();b=new.numpy()
        errors.append(dict(input=name,reference_l2=float(np.linalg.norm(a)),cached_l2=float(np.linalg.norm(b)),
            difference_l2=float(np.linalg.norm(a-b)),relative_difference=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))))
    timings={}
    for name,fn in (('reference_elastic_product',reference),('cached_elastic_product',cached),('refresh_cached_matrix',candidate.prepare)):
        fn();wp.synchronize_device(d)
        with wp.ScopedCapture(device=d) as capture:
            for _ in range(100):fn()
        wp.capture_launch(capture.graph);wp.synchronize_device(d);start=time.perf_counter()
        wp.capture_launch(capture.graph);wp.synchronize_device(d)
        timings[name]=(time.perf_counter()-start)/100
    result=dict(physics_steps=0,tick=sim.tick,block_nonzeros=candidate.nnz,block_storage_bytes=candidate.nnz*72,
        comparisons=errors,seconds_per_call=timings,
        note='Cached existing search matrix only; candidate is not wired into dynamics by this audit')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert errors[0]['relative_difference']<1e-10
    # Cancellation of huge elastic blocks is tested against the physical
    # inertial response to this translation, not a divide-by-zero old null.
    inertia=m.particle_mass.numpy()[:,None]*inputs[1]/(g.dt*g.dt)
    assert errors[1]['difference_l2']/np.linalg.norm(inertia)<1e-6

if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    from .config import Config
    from .simulation import Simulation
    from .provenance import record
    p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((a.recording/'config.json').read_text())
    s=Simulation(Config(**config))
    native=s.solver.native.native if hasattr(s.solver,'graphs') else s.solver.native
    g=native.global_refinement
    if g is None:raise ValueError('Recorded candidate must use the global search operator')
    with np.load(a.recording/'refinement_trace.npz',allow_pickle=False) as trace:
        available=np.flatnonzero(np.any(trace['q']!=0,axis=(1,2)))
        slot=int(available[-1]);g.q.assign(trace['q'][slot])
    record(a.output,s,dict(recording=str(a.recording),source_array='refinement_trace.npz:q',slot=slot,
        physics_steps=0,scope='Archived nonlinear geometry; elastic operator does not use contact caches or plastic rest angle'))
    audit(s,g,a.output/'result.json')

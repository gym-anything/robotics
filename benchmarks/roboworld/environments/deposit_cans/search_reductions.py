"""Same shell sums using block reductions instead of contended global atomics.

No force, matrix or tolerance is changed. The comparison must pass before this
path is selected in dynamics. Reference: installed Warp tile reduction examples.
"""
import warp as wp
from .relative_step import double
from .linear64 import matrix

@wp.struct
class Fields:
    x:wp.array[wp.vec3d]
    q:wp.array[wp.vec3]

@wp.func
def force_lane(lane:int,start:int,count:int,data:Fields):
    result=wp.vec3d(0.)
    for j in range(lane,count,128):result+=data.x[start+j]
    return result

@wp.func
def torque_lane(lane:int,start:int,count:int,data:Fields,center:wp.vec3):
    result=wp.vec3d(0.)
    for j in range(lane,count,128):
        i=start+j;result+=wp.cross(double(data.q[i])-double(center),data.x[i])
    return result

@wp.kernel
def sum_vectors(starts:wp.array[int],sizes:wp.array[int],x:wp.array[wp.vec3d],out:wp.array[wp.vec3d]):
    g=wp.tid();lanes=wp.tile_arange(128,dtype=int)
    data=Fields();data.x=x
    values=wp.tile_map(force_lane,lanes,starts[g],sizes[g],data)
    total=wp.tile_sum(values)*(wp.float64(1.)/wp.float64(sizes[g]))
    wp.tile_store(out,total,offset=(g,))

@wp.kernel
def restrict(q:wp.array[wp.vec3],starts:wp.array[int],sizes:wp.array[int],center:wp.array[wp.vec3],
    x:wp.array[wp.vec3d],force:wp.array[wp.vec3d],torque:wp.array[wp.vec3d]):
    g=wp.tid();lanes=wp.tile_arange(128,dtype=int)
    data=Fields();data.x=x;data.q=q
    values=wp.tile_map(force_lane,lanes,starts[g],sizes[g],data)
    moments=wp.tile_map(torque_lane,lanes,starts[g],sizes[g],data,center[g])
    wp.tile_store(force,wp.tile_sum(values),offset=(g,))
    wp.tile_store(torque,wp.tile_sum(moments),offset=(g,))

@wp.kernel
def internal_values(D:wp.array[wp.mat33],x:wp.array[wp.vec3d],groups:wp.array[int],
    mean:wp.array[wp.vec3d],out:wp.array[wp.vec3d]):
    i=wp.tid();g=groups[i];v=wp.vec3d(0.)
    if g>=0:v=wp.float64(4.)*matrix(D[i])*(x[i]-mean[g])
    out[i]=v

def audit(sim,g,path):
    import json,time
    import numpy as np
    m=g.s.model;d=m.device;l=g.linear;l.prepare_geometry();saved=l.tiled_reductions
    rng=np.random.default_rng(91);x=wp.array(rng.normal(size=(m.particle_count,3)),dtype=wp.vec3d,device=d)
    zero=wp.zeros_like(x);out=wp.zeros_like(x);rows=[];timings={}
    try:
        for name,fn in (('search_matrix',l.matvec),('preconditioner',l.precondition)):
            values=[]
            for enabled in (False,True):
                l.tiled_reductions=enabled;fn(x,zero,out,1.,0.);values.append(out.numpy())
                with wp.ScopedCapture(device=d) as cap:
                    for _ in range(100):fn(x,zero,out,1.,0.)
                wp.capture_launch(cap.graph);wp.synchronize_device(d);start=time.perf_counter()
                wp.capture_launch(cap.graph);wp.synchronize_device(d)
                timings[name+('_tiled' if enabled else '_atomic')]=(time.perf_counter()-start)/100
            a,b=values;error=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))
            rows.append(dict(operator=name,relative_difference=error,finite=bool(np.isfinite(values).all())))
    finally:l.tiled_reductions=saved
    result=dict(physics_steps=0,tick=sim.tick,comparisons=rows,seconds_per_call=timings,
        scope='Same actual recorded search matrices, different sum ordering only; not a full-runtime pass')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert all(r['finite'] and r['relative_difference']<1e-10 for r in rows)

"""Read-only containment of the actual deforming bag, not its bounding box.

The mouth cap exists only in this diagnostic mesh. It is never rendered or
passed to a collision pipeline. Native winding queries are explicitly enabled;
a topology/orientation check prevents their silent non-watertight fallback.
"""
from collections import Counter
import numpy as np
import warp as wp

@wp.kernel
def classify(mesh:wp.uint64,points:wp.array[wp.vec3],result:wp.array[int],distance:wp.array[float]):
    i=wp.tid();q=wp.mesh_query_point_sign_winding_number(mesh,points[i],20.,4.,.5)
    result[i]=0;distance[i]=20.
    if q.result:
        x=wp.mesh_eval_position(mesh,q.face,q.u,q.v)
        distance[i]=q.sign*wp.length(points[i]-x)
        result[i]=int(q.sign<0.)

def close_mouth(vertices,triangles,mouth):
    vertices=np.asarray(vertices,float);triangles=np.asarray(triangles,np.int32)
    mouth=np.asarray(mouth,np.int32)
    cap=np.array([(len(vertices),mouth[i],mouth[(i+1)%len(mouth)]) for i in range(len(mouth))],np.int32)
    v=np.vstack((vertices,vertices[mouth].mean(0)));f=np.vstack((triangles,cap))
    edges=Counter(tuple(sorted((int(t[i]),int(t[(i+1)%3])))) for t in f for i in range(3))
    directed=Counter((int(t[i]),int(t[(i+1)%3])) for t in f for i in range(3))
    if any(n!=2 for n in edges.values()) or any(directed[e]!=directed[e[::-1]] for e in directed):
        raise ValueError('Bag containment mesh is not closed and consistently oriented')
    volume=float(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6)
    if volume<=1e-5:raise ValueError('Invalid or reversed containment volume')
    return v,f,volume

class BagVolume:
    def __init__(self,sim):
        self.sim=sim;self.bag=sim.assembly.bag_spec;self.device=sim.model.device
        a=self.bag;sl=slice(a['first'],a['first']+a['count'])
        v,f,self.volume_m3=close_mouth(sim.state.particle_q.numpy()[sl],a.get('containment_triangles',a['triangles']),a['mouth'])
        self.points=wp.array(v,dtype=wp.vec3,device=self.device)
        self.faces=wp.array(f.ravel(),dtype=int,device=self.device)
        self._faces_host=f.copy()  # Immutable topology; positions remain live.
        self.mesh=wp.Mesh(points=self.points,indices=self.faces,support_winding_number=True)
        self.query=None;self.inside=None;self.distance=None
    def update(self,positions):
        a=self.bag;v=positions[a['first']:a['first']+a['count']]
        v=np.vstack((v,v[a['mouth']].mean(0)))
        f=self._faces_host
        area2=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)
        volume=float(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6)
        if not np.isfinite(v).all() or area2.min()<1e-10 or volume<=1e-5:
            raise ValueError('Invalid deformed containment mesh')
        self.points.assign(v.astype(np.float32));self.mesh.refit();self.volume_m3=volume
    def classify(self,points):
        p=np.asarray(points,np.float32).reshape(-1,3)
        if not np.isfinite(p).all():raise ValueError('Nonfinite containment query')
        if self.query is None or len(p)!=self.query.size:
            self.query=wp.empty(len(p),dtype=wp.vec3,device=self.device)
            self.inside=wp.empty(len(p),dtype=int,device=self.device)
            self.distance=wp.empty(len(p),dtype=float,device=self.device)
        self.query.assign(p)
        wp.launch(classify,len(p),[self.mesh.id,self.query,self.inside,self.distance],device=self.device)
        return self.inside.numpy().astype(bool),self.distance.numpy()

def self_check(device='cuda:0'):
    """Native geometry test with a deformed wall; no simulation advances."""
    from types import SimpleNamespace
    from .geometry import bag_mesh
    v,f,mouth,_=bag_mesh()
    data=wp.array(v,dtype=wp.vec3,device=device)
    a=dict(first=0,count=len(v),triangles=f,mouth=mouth)
    sim=SimpleNamespace(model=SimpleNamespace(device=device),assembly=SimpleNamespace(bag_spec=a),
        state=SimpleNamespace(particle_q=data))
    volume=BagVolume(sim)
    p=np.array([[0,0,.5],[.4,0,.5],[0,0,1.1],[0,0,0.],[.24,0,.5]])
    initial,_=volume.classify(p)
    # Move the entire positive-x wall inward below the cap. A fixed box test
    # would still call the last point inside; the actual deformed mesh must not.
    changed=v.copy();changed[:,0]*=.5
    volume.update(changed);deformed,_=volume.classify(p)
    assert initial.tolist()==[True,False,False,False,True],initial
    assert deformed.tolist()==[True,False,False,False,False],deformed
    return dict(physics_steps=0,initial=initial.tolist(),deformed=deformed.tolist())

if __name__=='__main__':
    import json
    print(json.dumps(self_check()))

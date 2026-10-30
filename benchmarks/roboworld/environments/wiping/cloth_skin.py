"""Visible cloth envelope from its existing physical contact half-thickness.

This offsets the recorded center surface by +/- particle radius and closes only
its boundary. No smoothing, alternative cloth trajectory or extra physics step.
The center mesh remains the public deformable state and collision discretization.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=8)
def _topology(raw,count):
    triangles=np.frombuffer(raw,dtype=np.int32).reshape(-1,3);edges={}
    for tri in triangles:
        for a,b in zip(tri,np.roll(tri,-1)):
            key=tuple(sorted((int(a),int(b))))
            if key in edges:edges.pop(key)
            else:edges[key]=(int(a),int(b))
    sides=[]
    for a,b in edges.values():sides.extend(((a+count,b+count,b),(a+count,b,a)))
    return np.concatenate((triangles,triangles[:,[0,2,1]]+count,np.asarray(sides,np.int32)))


def geometry(points,triangles,radius):
    from .render import normals
    tri=np.asarray(triangles,np.int32);normal=normals(points,tri)
    vertices=np.concatenate((points+radius*normal,points-radius*normal)).astype(np.float32)
    return vertices,_topology(tri.tobytes(),len(points))


if __name__=='__main__':
    import argparse,json
    from collections import Counter
    from pathlib import Path
    from .assembly import folded_grid
    from .config import Config
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=Config();_,points,tri=folded_grid(c,(0,0,.9));points=points.astype(np.float32)
    skin,faces=geometry(points,tri,c.cloth_radius);n=len(points)
    edges=Counter(tuple(sorted((int(row[i]),int(row[(i+1)%3])))) for row in faces for i in range(3))
    error=float(abs((skin[:n]+skin[n:])/2-points).max())
    radius_error=float(abs(np.linalg.norm(skin[:n]-points,axis=1)-c.cloth_radius).max())
    result=dict(watertight=all(v==2 for v in edges.values()),center_surface_max_error_m=error,
        half_thickness_max_error_m=radius_error,physics_steps=0,center_vertices=n,skin_vertices=len(skin),
        scope='Topology/offset check only; visual inspection remains required')
    assert result['watertight'] and max(error,radius_error)<1e-7
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

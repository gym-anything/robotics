"""Formed open GN pan with one shared optical/collision shell.

Manufacturer envelope/flange; authored 3 mm wall and corner/taper profile.
Flat physical patches and triangulated corner strips are explicitly mapped for
cleaning. No decorative shell covers different physical geometry.
"""
import numpy as np
import newton
import warp as wp
from .room_fixtures import rounded_ring


def geometry(size):
    x,y,h=size;hx=x/2;hy=y/2;r=.024;t=.003
    # All concentric contours share their corner centers, so each straight
    # wall/rim strip is an exact rectangle (including the sloping walls).
    specs=((0.,h),(.0127,h),(.019,.003),(.016,0.),(.0097,h-t),(0.,h-t))
    rings=[rounded_ring(hx-d,hy-d,r-d,z) for d,z in specs]
    n=len(rings[0]);vertices=np.concatenate(rings);triangles=[];patches=[]
    for a,b in ((0,1),(1,2),(3,4),(4,5),(5,0)):
        for i in range(n):
            j=(i+1)%n
            triangles.extend(((a*n+i,a*n+j,b*n+j),(a*n+i,b*n+j,b*n+i)))
            if i%9==8:
                p=rings[a][i];u=rings[a][j]-p;v=rings[b][i]-p
                lu,lv=np.linalg.norm(u),np.linalg.norm(v)
                if abs(float(u@v))>1e-8:raise ValueError('Nonrectangular mapped strip')
                patches.append(dict(name=f'strip_{a}_{b}_{i}',origin=p,u=u/lu,v=v/lv,size=(lu,lv)))
            else:
                for k,ids in enumerate(((a*n+i,a*n+j,b*n+j),(a*n+i,b*n+j,b*n+i))):
                    patches.append(dict(name=f'corner_{a}_{b}_{i}_{k}',triangle=vertices[list(ids)]))
    for i in range(1,n-1):
        triangles.extend(((2*n,2*n+i,2*n+i+1),(3*n,3*n+i+1,3*n+i)))
    # Nonoverlapping cross plus four triangulated sectors covers the whole floor.
    for ring,sign in ((2,1),(3,-1)):
        d,z=specs[ring];ax,ay,rr=hx-d,hy-d,r-d
        pieces=[(-ax+rr,-ay,2*(ax-rr),2*ay),
                (-ax,-ay+rr,rr,2*(ay-rr)),(ax-rr,-ay+rr,rr,2*(ay-rr))]
        for k,(px,py,w,l) in enumerate(pieces):
            patches.append(dict(name=f'floor_{ring}_{k}',origin=np.array((px,py if sign==1 else py+l,z)),
                u=np.array((1.,0.,0.)),v=np.array((0.,float(sign),0.)),size=(w,l)))
        for corner in range(4):
            arc=rings[ring][corner*9:corner*9+9]
            center=np.array(((1 if corner in (0,3) else -1)*(ax-rr),
                             (1 if corner in (0,1) else -1)*(ay-rr),z))
            for i in range(8):
                points=np.array((center,arc[i],arc[i+1]))
                if sign<0:points=points[[0,2,1]]
                patches.append(dict(name=f'floor_corner_{ring}_{corner}_{i}',triangle=points))
    return vertices,np.asarray(triangles,np.int32).ravel(),patches


def build(a,name,center,size,mass,material):
    from .kitchen import set_mass
    b=a.builder;body=b.add_body(xform=wp.transform(center,wp.quat_identity()),label=name)
    vertices,indices,patches=geometry(size)
    mesh=newton.Mesh(vertices,indices)
    if not a.render_only:mesh.build_sdf(device=a.config.device,max_resolution=256)
    shape=b.add_shape_mesh(body,mesh=mesh,cfg=a.shape_config(),label=name+'_formed_shell')
    a.materials[shape]=material;set_mass(b,body,mass)
    a.objects[name]=dict(body=body,bottom_shape=shape,size=size,wall=.003,mass=mass,
        geometry='formed open rounded shell; shared collision/render mesh',
        planar_patches=[{k:(v if k=='name' else np.asarray(v).tolist()) for k,v in p.items()} for p in patches],
        corner_handling='triangle material cells on the actual collision facets',
        outer_corner_radius=.024,flange_width=.0127,bottom_inset=.019)
    return body


if __name__=='__main__':
    import argparse,json
    from collections import Counter
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();results=[]
    for size in ((.265,.325,.065),(.176,.325,.065)):
        v,f,patches=geometry(size);tri=f.reshape(-1,3)
        edge=Counter(tuple(sorted((int(row[i]),int(row[(i+1)%3])))) for row in tri for i in range(3))
        volume=float(np.einsum('ij,ij->i',v[tri[:,0]],np.cross(v[tri[:,1]],v[tri[:,2]])).sum()/6)
        area=np.linalg.norm(np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]]),axis=1).sum()/2
        mapped=sum(float(np.linalg.norm(np.cross(patch['triangle'][1]-patch['triangle'][0],patch['triangle'][2]-patch['triangle'][0]))/2)
            if 'triangle' in patch else float(np.prod(patch['size'])) for patch in patches)
        assert all(count==2 for count in edge.values()) and volume>0
        assert np.allclose(v.max(0)-v.min(0),size,atol=1e-7)
        assert np.isclose(mapped,area,rtol=1e-5)
        results.append(dict(size=size,watertight=True,positive_volume_m3=volume,triangles=len(tri),
            surface_area_m2=float(area),mapped_area_m2=mapped,unmapped_curved_fraction=1-mapped/float(area),
            assistance='Geometry audit only; zero simulation ticks, not a physical contact pass'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))

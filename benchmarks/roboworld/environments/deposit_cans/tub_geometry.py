"""Authored Cambox-style molded tub, one watertight collision/render surface.

Manufacturer exterior envelope; rim, grip recesses, ribs and wall thickness
are reconstructed from the selected 1520CBP product image, not measured CAD.
No invented loop handles. The underside of the real rim provides purchase.
"""
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.path import Path as PolygonPath
from wiping.room_fixtures import rounded_ring

SOURCE='https://cambro.widen.net/content/ra5k9t1zj2/webp/1520CBP131_A1C0_0718_S01.webp'

def geometry(size=(.389,.506,.125),*,thickness=.0017):
    x,y,h=size;hx=x/2;hy=y/2;r=.024;t=thickness
    if not .001<=t<=.003:raise ValueError('Tub wall estimate outside1..3mm')
    # Skin descends at the outer rim to form its molded return/apron. A
    # cavity remains beneath the flange: not a solid thick plastic ledge.
    specs=((0.,h),(.0127,h),(.019,t),(.019-t,0.),
           (.0127-t,h-t),(t,h-t),(t,h-.013),(0.,h-.013))
    rings=[rounded_ring(hx-d,hy-d,r-d,z) for d,z in specs]
    # The reusable rounded-ring helper samples corners, but its straight
    # sections contain endpoints only. Resolve the grip region explicitly;
    # otherwise evaluating its profile at those endpoints creates NO recess.
    resolved=[]
    for ring in rings:
        points=[]
        for i,a in enumerate(ring):
            b=ring[(i+1)%len(ring)];points.append(a.copy())
            if abs(b[0]-a[0])>.20 and abs(b[1]-a[1])<1e-8:
                xs=np.array([-.09,-.075,-.065,0.,.065,.075,.09])
                if b[0]<a[0]:xs=xs[::-1]
                points.extend(np.array((x,a[1],a[2]))for x in xs)
        resolved.append(np.asarray(points))
    rings=resolved
    for j,ring in enumerate(rings):
        if j in (2,3):continue
        # Shallow thumb/grip area on the two short ends, joined continuously
        # to the unchanged rim. Three millimetres is an authored estimate.
        weight=np.clip((.090-np.abs(ring[:,0]))/.025,0.,1.)
        short=np.isclose(np.abs(ring[:,1]),hy-specs[j][0],atol=1e-9)
        ring[:,2]-=.003*weight*short
    n=len(rings[0]);vertices=np.concatenate(rings);faces=[]
    for a,b in ((0,1),(1,2),(3,4),(4,5),(5,6),(6,7),(7,0)):
        for i in range(n):
            j=(i+1)%n;faces.extend(((a*n+i,a*n+j,b*n+j),(a*n+i,b*n+j,b*n+i)))
    # Triangulated ribs occur in BOTH sides of the bottom skin. They change
    # real contact geometry; their visible ridges are not normal-map bumps.
    for ring,sign in ((2,1),(3,-1)):
        boundary=rings[ring];d,z=specs[ring]
        grid=np.array([(xx,yy)for yy in np.arange(-hy+d+.006,hy-d,.008)
                       for xx in np.arange(-hx+d+.006,hx-d,.008)])
        grid=grid[PolygonPath(boundary[:,:2]).contains_points(grid)]
        # Avoid slivers against the closely sampled rounded perimeter.
        for i,a in enumerate(boundary[:,:2]):
            b=boundary[(i+1)%n,:2];ab=b-a
            u=np.clip((grid-a)@ab/(ab@ab),0.,1.)
            grid=grid[np.linalg.norm(grid-(a+u[:,None]*ab),axis=1)>.003]
        ids=np.r_[np.arange(ring*n,(ring+1)*n),np.arange(len(vertices),len(vertices)+len(grid))]
        inside=np.column_stack((grid,np.full(len(grid),z)))
        vertices=np.vstack((vertices,inside))
        xy=vertices[ids,:2]
        # Broad molded stiffening ridges with rounded longitudinal ends;
        # 3mm rise,25mm width and160mm length are disclosed estimates.
        lift=np.zeros(len(xy));long=np.clip((.095-np.abs(xy[:,1]))/.020,0.,1.)
        long=long*long*(3.-2.*long)
        for center in (-.10,0.,.10):
            cross=np.clip(1.-np.abs(xy[:,0]-center)/.0125,0.,1.)
            lift+=.003*long*cross*cross*(3.-2.*cross)
        vertices[ids,2]+=lift
        tri=ids[Delaunay(xy).simplices]
        faces.extend((tri if sign==1 else tri[:,::-1]).tolist())
    return vertices,np.asarray(faces,np.int32)

def audit():
    from collections import Counter
    v,f=geometry();p=v[f]
    edge=Counter(tuple(sorted((int(t[i]),int(t[(i+1)%3]))))for t in f for i in range(3))
    oriented=Counter((int(t[i]),int(t[(i+1)%3]))for t in f for i in range(3))
    assert all(n==2 for n in edge.values()),'Tub must be watertight'
    assert all(oriented[b,a]==n for (a,b),n in oriented.items()),'Inconsistent winding'
    area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)/2
    assert area.min()>1e-10,'Degenerate tub facet'
    volume=float(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6)
    assert volume>0
    assert np.allclose(v.max(0)-v.min(0),(.389,.506,.125),atol=1e-8)
    grips=(np.abs(v[:,0])<.066)&(np.abs(np.abs(v[:,1])-.253)<1e-6)&(v[:,2]>.12)
    assert grips.sum()>=6,'No sampled rim grip region'
    assert np.allclose(v[grips,2],.122,atol=1e-7),'Grip recess not represented'
    # CPU-only intersection check, no rigid pose update or simulation ticks.
    import ipctk
    edges=np.unique(np.sort(np.vstack((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1),axis=0)
    surface=ipctk.CollisionMesh(v,edges,f)
    crossing=bool(ipctk.has_intersections(surface,v))
    assert not crossing,'Self-intersecting molded tub'
    return dict(vertices=len(v),triangles=len(f),watertight=True,oriented=True,
        self_intersection=crossing,volume_m3=volume,envelope_m=(v.max(0)-v.min(0)).tolist(),
        minimum_triangle_area_m2=float(area.min()),fixed_mass_kg=.70,
        geometry_density_kg_m3=.70/volume,
        source=SOURCE,estimated_features='1.7mm skin;13mm return lip;3mm thumb recess;3mm bottom ribs',
        physics_steps=0,qualification='Geometry only; new contact and tub-lift qualification required')

if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);result=audit()
    v,f=geometry();np.savez_compressed(a.output/'tub.npz',vertices=v,faces=f)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

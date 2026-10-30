"""Collision-bearing, off-task handwash station for the commercial prep room.

Manufacturer envelope references are in SCENE_SOURCES.md. Basin profile,
gooseneck and installation positions are authored. The tap is off; no hidden
water/soap source or new policy action is implied by these room fixtures.
"""
import math
import numpy as np
import newton
import warp as wp

def rounded_ring(hx,hy,r,z):
    return np.asarray([(sx*(hx-r)+r*math.cos(t),sy*(hy-r)+r*math.sin(t),z)
        for sx,sy,start in ((1,1,0),(-1,1,90),(-1,-1,180),(1,-1,270))
        for t in np.deg2rad(np.linspace(start,start+90,9))],np.float32)

def basin_geometry():
    # Six rounded rectangular rings; no closed invisible box inside the bowl.
    rings=[rounded_ring(.216,.200,.015,.900),rounded_ring(.170,.1525,.030,.899),
           rounded_ring(.143,.1275,.025,.758),rounded_ring(.146,.1305,.028,.755),
           rounded_ring(.173,.1555,.033,.896),rounded_ring(.216,.200,.015,.897)]
    n=len(rings[0]);v=np.concatenate(rings);f=[]
    for a,b in ((0,1),(1,2),(3,4),(4,5),(5,0)):
        for i in range(n):
            j=(i+1)%n;f.extend(((a*n+i,a*n+j,b*n+j),(a*n+i,b*n+j,b*n+i)))
    for i in range(1,n-1):
        f.append((2*n,2*n+i,2*n+i+1))
        f.append((3*n,3*n+i+1,3*n+i))
    return v,np.asarray(f,np.int32).ravel()

def build(a):
    b=a.builder;v,f=basin_geometry();mesh=newton.Mesh(v,f)
    if not a.render_only:mesh.build_sdf(device=a.config.device,max_resolution=256)
    shape=b.add_shape_mesh(-1,xform=wp.transform((1.157,1.12,0),wp.quat_identity()),
        mesh=mesh,cfg=a.shape_config(density=0.),label='handwash_basin')
    a.materials[shape]='steel'
    a.box('sink_back_mount',(1.359,1.12,.925),(.014,.2,.075),'steel')
    # Wall shroud covers the plumbing, with a physically present fixed envelope.
    a.box('sink_trap_shroud',(1.28,1.12,.65),(.08,.14,.105),'steel')
    def tube(name,p0,p1,r,mat='steel'):
        p0=np.asarray(p0);p1=np.asarray(p1);d=p1-p0;length=float(np.linalg.norm(d));d/=length
        if d[2]<-.999999:q=wp.quat_from_axis_angle(wp.vec3(1.,0.,0.),math.pi)
        else:
            w=math.sqrt((1+d[2])/2);q=wp.quat(-d[1]/(2*w),d[0]/(2*w),0.,w)
        s=b.add_shape_capsule(-1,xform=wp.transform((p0+p1)/2,q),radius=r,half_height=length/2,
            cfg=a.shape_config(density=0.),label=name)
        a.materials[s]=mat
    tube('tap_riser',(1.329,1.12,.918),(1.329,1.12,1.12),.012)
    points=[(1.279+.05*math.cos(t),1.12,1.12+.05*math.sin(t)) for t in np.linspace(0,math.pi,13)]
    for i in range(len(points)-1):tube(f'tap_bend_{i}',points[i],points[i+1],.012)
    tube('tap_spout',points[-1],(1.229,1.12,1.075),.012)
    tube('tap_aerator',(1.229,1.12,1.075),(1.229,1.12,1.061),.015,'charcoal')
    tube('tap_lever',(1.335,1.06,.948),(1.27,1.06,.975),.006)
    # Bobrick B-2111-sized closed dispenser; valve is deliberately inactive.
    a.box('soap_dispenser',(1.337,.80,1.29),(.035,.060,.1025),'steel')
    a.box('soap_level_window',(1.301,.80,1.275),(.001,.006,.018),'charcoal')
    tube('soap_push_button',(1.302,.80,1.221),(1.282,.80,1.221),.013,'charcoal')
    # Folded-paper dispenser envelope; no fake graspable hanging paper.
    a.box('paper_dispenser',(1.308,1.48,1.43),(.064,.14,.16),'steel')
    a.box('paper_aperture',(1.242,1.48,1.292),(.003,.105,.009),'charcoal')

if __name__=='__main__':
    # Topology sanity only, not visual/physics qualification of the room.
    from collections import Counter
    import json
    v,f=basin_geometry();tri=f.reshape(-1,3)
    edges=Counter(tuple(sorted((int(row[i]),int(row[(i+1)%3])))) for row in tri for i in range(3))
    volume=float(np.sum(np.einsum('ij,ij->i',v[tri[:,0]],np.cross(v[tri[:,1]],v[tri[:,2]])))/6)
    area=np.linalg.norm(np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]]),axis=1)/2
    assert all(count==2 for count in edges.values()),'Open/nonmanifold basin shell'
    assert volume>0 and area.min()>0,'Inverted or degenerate basin shell'
    print(json.dumps(dict(watertight=True,volume_m3=volume,minimum_triangle_area_m2=float(area.min()),
        extents_m=(v.max(0)-v.min(0)).tolist(),triangles=len(tri))))

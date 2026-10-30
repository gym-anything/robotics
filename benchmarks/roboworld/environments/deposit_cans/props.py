"""Metric movable preservation props and wrong-destination waste container.

Manufacturer envelopes are sourced; wall profiles, masses of unlisted pieces
and installation positions are explicit authored estimates, not measurements.
All visible prop envelopes here are also colliders, with genuinely open bowls.
"""
import numpy as np
import newton
import warp as wp
from wiping.room_fixtures import rounded_ring

def rounded_shell(size,wall=.003,inset=.025,radius=.040):
    x,y,h=size;hx=x/2;hy=y/2
    specs=((hx,hy,radius,h),(hx-inset,hy-inset,max(radius-inset,.008),0.),
        (hx-inset-wall,hy-inset-wall,max(radius-inset-wall,.005),wall),
        (hx-wall,hy-wall,radius-wall,h))
    rings=[rounded_ring(*row) for row in specs];n=len(rings[0]);v=np.concatenate(rings);f=[]
    for a,b in ((0,1),(2,3),(3,0)):
        for i in range(n):
            j=(i+1)%n;f.extend(((a*n+i,a*n+j,b*n+j),(a*n+i,b*n+j,b*n+i)))
    # Side normals of CCW rings point inward for descending edges: reverse
    # all side faces. Bottom disks use their corresponding outward winding.
    f=[tuple(reversed(t)) for t in f]
    for i in range(1,n-1):f.extend(((n,n+i+1,n+i),(2*n,2*n+i,2*n+i+1)))
    return v,np.asarray(f,np.int32)

# sort_waste's single-use paper hot cup, 12 oz class: rim 90 mm, base 60 mm, height 110 mm outer; the 1.5 mm
# wall and floor and the 12 g mass are authored estimates (task.PAPER_CUP_* is this outer frustum).
PAPER_CUP_RINGS=((.045,.110),(.030,0.),(.0285,.0015),(.0435,.110))
PAPER_CUP_MASS=.012

def cup_geometry(n=48,rings_rz=((.0375,.132),(.028,0.),(.026,.003),(.0355,.132))):
    # Default: Cambro 1200CW outer 75x132mm; tapered PC walls/heel authored. Rings: outer rim, outer base,
    # inner base, inner rim (radius, height).
    rings=[];theta=np.arange(n)*2*np.pi/n
    for r,z in rings_rz:
        rings.append(np.c_[r*np.cos(theta),r*np.sin(theta),np.full(n,z)])
    v=np.concatenate(rings);f=[]
    for a,b in ((0,1),(2,3),(3,0)):
        for i in range(n):
            j=(i+1)%n;f.extend(((a*n+i,b*n+j,a*n+j),(a*n+i,b*n+i,b*n+j)))
    for i in range(1,n-1):f.extend(((n,n+i+1,n+i),(2*n,2*n+i,2*n+i+1)))
    return v,np.asarray(f,np.int32)

def mesh_body(a,name,position,v,f,material,mass,resolution=256):
    from collections import Counter
    pairs=Counter(tuple(sorted((int(row[i]),int(row[(i+1)%3])))) for row in f for i in range(3))
    volume=float(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6)
    if any(count!=2 for count in pairs.values()) or volume<=0:raise ValueError('Invalid closed prop mesh: '+name)
    b=a.builder;body=b.add_body(xform=wp.transform(position,wp.quat_identity()),label=name)
    mesh=newton.Mesh(v,f.ravel());mesh.build_sdf(device=a.config.device,max_resolution=resolution)
    shape=b.add_shape_mesh(body,mesh=mesh,cfg=a.shape_config(density=mass/volume),label=name+'_shell')
    a.materials[shape]=material;a.set_mass(body,mass)
    a.objects[name]=dict(body=body,mass_kg=mass,envelope_m=(v.max(0)-v.min(0)).tolist(),
        physical_geometry='open hollow watertight shell',volume_m3=volume)
    return body

def build(a):
    v,f=rounded_shell((.40,.18,.020),wall=.002,inset=.004,radius=.015)
    tray=mesh_body(a,'reusable_return_tray',(-.43,.375,.903),v,f,'gray_plastic',.23)
    from .variants import preset,SORT_CUPS_ON_COUNTER,TUB_CUPS,PAPER_CUPS
    # The cups start on the tray, except sort_cups (on the counter's right-hand row, between counter cans) and
    # tub_cups (standing in the tub); the tray then stays empty in its usual place.
    task=preset(a.config);starts={'sort_cups':SORT_CUPS_ON_COUNTER,'tub_cups':TUB_CUPS}.get(task)
    for i,x in enumerate((-.56,-.43,-.30)):
        v,f=cup_geometry()
        mesh_body(a,f'reusable_cup_{i}',starts[i] if starts else (x,.375,.909),v,f,'polycarbonate',.077)
    # Rubbermaid FG354060: body+end handles reach 558.8x279.4x762mm.
    v,f=rounded_shell((.508,.2794,.762),wall=.004,inset=.030,radius=.045)
    waste=mesh_body(a,'landfill_container',(.50,.99,.001),v,f,'charcoal',3.54,resolution=512)
    for sign in (-1,1):
        a.box('waste_handle_'+str(sign),(sign*.2667,0,.714),(.0127,.075,.012),'charcoal',waste)
    a.set_mass(waste,3.54)
    a.objects['landfill_container'].update(envelope_m=[.5588,.2794,.762],
        body_envelope_m=[.508,.2794,.762],manufacturer='Rubbermaid FG354060BLA')
    a.objects['preservation']=dict(tray=tray,cups=[a.objects[f'reusable_cup_{i}']['body'] for i in range(3)])
    if task=='sort_waste':
        # Two used paper cups on the counter, for the landfill bin. Shared prop contact configuration.
        v,f=cup_geometry(rings_rz=PAPER_CUP_RINGS)
        a.objects['paper_cups']=[mesh_body(a,f'paper_cup_{k}',position,v,f,'paper_cup',PAPER_CUP_MASS)
            for k,position in enumerate(PAPER_CUPS)]

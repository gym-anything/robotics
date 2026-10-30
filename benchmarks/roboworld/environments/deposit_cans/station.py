"""Assembled folding recycling stand and hollow molded slotted lid.

Published: overall40x25x18in,3/8in rod,8lb total,1.75lb lid,16x4.5in slot.
The manufacturer's installation photographs show crossed folding legs,
upper bag-lock loops, a raised skirted lid and four pairs of underside clips.
Unpublished radii, molding detail and pivot placement are authored estimates.
The station is installed and locked before an episode, not a folding task.
"""
import numpy as np
import newton
import warp as wp
from wiping.room_fixtures import rounded_ring

def lid_mesh(revision=2):
    # One continuous wall around the slot and outer skirt; never fill the slot
    # with a convex collision hull. Coordinates are local to the frame base.
    specs=[(.3175,.2286,.025,.966),(.3175,.2286,.025,.975),
        (.2835,.1946,.022,1.016),(.2032,.05715,.015,1.016),
        (.2032,.05715,.015,1.013),(.2821,.1932,.020,1.013),
        (.3145,.2256,.022,.974),(.3145,.2256,.022,.966)]
    if revision>=3:
        # The earlier roof narrowed through its own support rods and bag locks.
        # Preserve published outer envelope/slot/wall; these unpublished roof
        # shoulders now enclose the actual assembled rods and folded cuff.
        specs[0]=(.3175,.2286,.018,.966)
        specs[1]=(.3175,.2286,.018,.975)
        specs[2]=(.316,.2276,.015,1.016)
        specs[5]=(.3146,.2262,.014,1.013)
        specs[6]=(.3145,.2256,.014,.974)
        specs[7]=(.3145,.2256,.014,.966)
    rings=[rounded_ring(*row) for row in specs];n=len(rings[0]);v=np.concatenate(rings);f=[]
    for a in range(len(rings)):
        b=(a+1)%len(rings)
        for i in range(n):
            j=(i+1)%n;f.extend(((a*n+i,a*n+j,b*n+j),(a*n+i,b*n+j,b*n+i)))
    f=np.asarray(f,np.int32)
    volume=float(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6)
    if volume<0:f=f[:,::-1];volume=-volume
    from collections import Counter
    count=Counter(tuple(sorted((int(t[i]),int(t[(i+1)%3])))) for t in f for i in range(3))
    if not volume>0 or any(n!=2 for n in count.values()):raise ValueError('Invalid molded lid shell')
    return v,f,volume

def build(a,center):
    b=a.builder;body=b.add_body(xform=wp.transform(center,wp.quat_identity()),label='recycling_frame')
    radius=.0047625
    # Two offset U frames cross at the riveted pivots. The side Xs are outside
    # the bag envelope, not bars crossing through the loaded bag interior.
    for frame in (0,1):
        hx=.2965+frame*.0098;top_y=(-1 if frame else 1)*.2126
        a.tube(f'frame_{frame}_top',(-hx,top_y,.989),(hx,top_y,.989),radius,body)
        for sign in (-1,1):
            x=sign*hx
            a.tube(f'frame_{frame}_leg_{sign}',(x,-top_y,.011),(x,top_y,.989),radius,body)
            a.box(f'frame_{frame}_foot_{sign}',(x,-top_y,.006),(.007,.011,.006),
                'rubber',body,cfg=a.shape_config(density=1100.,mu=.65))
        # Narrow upper loops carry the installed cuff and stop it sliding off.
        for sign in (-1,1):
            x=sign*(hx-.019)
            points=[(x,top_y,.989),(x,top_y*.91,1.002),
                (x-sign*.075,top_y*.91,1.002),(x-sign*.075,top_y,.989)]
            for j,(p,q) in enumerate(zip(points,points[1:])):
                a.tube(f'bag_lock_{frame}_{sign}_{j}',p,q,.0032,body)
    for sign in (-1,1):
        a.tube('pivot_'+str(sign),(sign*.288,0,.500),(sign*.314,0,.500),.0075,body)
    a.set_mass(body,3.62874-.79378665)
    frame_mass=b.body_mass[body];frame_com=np.asarray(b.body_com[body]).copy()
    frame_inertia=np.asarray(b.body_inertia[body]).reshape(3,3).copy()
    # Add all molding pieces at a common estimated polymer density, then scale
    # only their combined mass/inertia to the published lid mass below.
    v,f,volume=lid_mesh(a.config.scene_revision);mesh=newton.Mesh(v,f.ravel())
    mesh.build_sdf(device=a.config.device,max_resolution=512)
    shape=b.add_shape_mesh(body,mesh=mesh,cfg=a.shape_config(density=950.),label='rounded_slot_lid')
    a.materials[shape]='blue_plastic'
    for y in (-.211,.211):
        for x in (-.19,.19):
            # Pair of molded retaining lugs around a top cross rod.
            for offset in (-.0068,.0068):
                a.box(f'lid_clip_{x}_{y}_{offset}',(x,y+offset,.989),(.009,.002,.011),
                    'blue_plastic',body,cfg=a.shape_config(density=950.))
    for y in (-.177,.177):
        a.box('lid_rib_'+str(y),(0,y,1.005),(.252,.0015,.006),
            'blue_plastic',body,cfg=a.shape_config(density=950.))
    # Builder inertia is about the compound COM. Separate its frame contribution
    # at the origin before scaling the lid contribution, then restore COM form.
    def shift(c):return np.eye(3)*np.dot(c,c)-np.outer(c,c)
    total_mass=b.body_mass[body];total_com=np.asarray(b.body_com[body])
    total_origin=np.asarray(b.body_inertia[body]).reshape(3,3)+total_mass*shift(total_com)
    frame_origin=frame_inertia+frame_mass*shift(frame_com)
    raw_lid_mass=total_mass-frame_mass;factor=.79378665/raw_lid_mass
    lid_first=total_mass*total_com-frame_mass*frame_com
    mass=frame_mass+.79378665;com=(frame_mass*frame_com+factor*lid_first)/mass
    inertia=frame_origin+factor*(total_origin-frame_origin)-mass*shift(com)
    if np.linalg.eigvalsh(inertia).min()<=0:raise ValueError('Invalid station inertia')
    b.body_mass[body]=mass;b.body_com[body]=wp.vec3(*com)
    b.body_inertia[body]=wp.mat33(*inertia.ravel().tolist())
    a.objects['station']=dict(body=body,slot_m=[.4064,.1143],height_m=1.016,mass_kg=mass,
        lid_mass_kg=.79378665,lid_wall_m=.003,slot_corner_radius_m=.015,
        construction='Installed crossed-U stand, locked hollow molded lid and four paired clips; unpublished details authored')
    return body

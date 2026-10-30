"""Solid cable guides and read-only centreline routing measurements.

Guide dimensions are authored workshop-accessory estimates, not a claimed
Rittal/Phoenix product replica. Contact, gravity and friction supply support.
Passing through a guide creates no joint, snap, attractor or electrical edge.
"""
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from .geometry import annulus_surface

def install(assembly):
    assembly.routing_guides={}
    # Closed, radiused 22 mm-bore eyelets. Both connector types can pass before
    # mating. Brackets fix the eyes to the actual module/mounting plate.
    for name,cable,centre,mount_x in (
        ("G1","sensor",(.530,-.220,1.025),.653),
        ("G2","network",(.514,.166,1.100),.568),
        ("G3","supply_brown",(.535,-.015,.918),.653),
        ("G4","supply_blue",(.535,-.015,.875),.653)):
        rotation=wp.quat_from_axis_angle(wp.vec3(1,0,0),np.pi/2)
        frame=wp.transform(centre,rotation)
        # A 0.5 mm bevel prevents a sharp sheet-metal edge at the cable jacket.
        def inner(z,a):return .011+max(0.,abs(z)-.0025)
        mesh=annulus_surface([-.003,-.0025,.0025,.003],inner,
                             lambda z,a:.014,sectors=64)
        shape=assembly._mesh(name+"_guide_eye",-1,mesh,frame,sdf=True,
            material="nickel",cfg=assembly.shape_config(ke=60000,kd=1.,mu=.35,gap=.00005))
        x,y,z=centre
        assembly.box(name+"_support",((x+.013+mount_x)/2,y,z),
            ((mount_x-x-.013)/2,.004,.003),material="nickel")
        assembly.box(name+"_mount",(mount_x,y,z),(.002,.012,.015),material="nickel")
        for dz in (-.009,.009):
            assembly._static_cylinder(name+"_bolt",(mount_x-.003,y,z+dz),.0018,.003,
                "nickel",rotation=wp.quat_from_axis_angle(wp.vec3(0,1,0),np.pi/2))
        assembly.label(name,(x-.016,y,z+.020),.018,.008)
        assembly.routing_guides[name]=dict(cable=cable,frame=np.asarray(frame).tolist(),
            inner_radius_m=.011,depth_m=.006,bevel_m=.0005,shape_id=shape,
            support="closed radiused steel eye; permanent bolted bracket")

def rod_segments(assembly,q,cable):
    ids=assembly.objects[cable+"_cable"]
    # A curved authored centreline is discretized into unequal chords.
    # length/segments is NOT their exact capsule centreline length. Reading
    # the actual collision primitives avoids synthetic joint gaps/endpoints.
    import newton
    b=assembly.builder
    if not hasattr(assembly,"_routing_half_lengths"):
        assembly._routing_half_lengths={}
    if cable not in assembly._routing_half_lengths:
        half=[]
        for body in ids:
            shapes=[s for s in b.body_shapes[body] if b.shape_type[s]==newton.GeoType.CAPSULE]
            if len(shapes)!=1:raise ValueError("Expected one physical rod capsule per segment")
            shape=shapes[0]
            if np.linalg.norm(np.asarray(b.shape_transform[shape])[:3])>1e-8:
                raise ValueError("Routing requires COM-centred physical rods")
            half.append(float(b.shape_scale[shape][1]))
        assembly._routing_half_lengths[cable]=np.asarray(half)
    h=assembly._routing_half_lengths[cable]
    pose=q[ids]
    offset=Rotation.from_quat(pose[:,3:]).apply(np.column_stack((np.zeros(len(ids)),np.zeros(len(ids)),h)))
    return pose[:,:3]-offset,pose[:,:3]+offset

def crossings(segments,frame,clear_radius,depth,*,bevel=0.,signed=False):
    """Require a physical rod centreline spanning both faces of the bore.

    Include neighbouring segment end gaps only through the independently
    measured rod-joint continuity check in inspect(); no synthetic long line.
    """
    a,b=segments
    r=Rotation.from_quat(frame[3:])
    a=r.inv().apply(a-np.asarray(frame[:3]))
    b=r.inv().apply(b-np.asarray(frame[:3]))
    # Rods meeting at the middle plane require contiguous adjacent segments;
    # sample actual endpoints in chain order, not an endpoint-to-endpoint chord.
    points=np.empty((2*len(a),3));points[0::2]=a;points[1::2]=b
    hits=[]
    for i in range(len(points)-1):
        p,s=points[i],points[i+1]
        if p[2]*s[2]>0 or abs(s[2]-p[2])<1e-12:continue
        u=-p[2]/(s[2]-p[2]);xy=p[:2]+u*(s[:2]-p[:2])
        if np.linalg.norm(xy)>clear_radius:continue
        # Follow both directions to the bore faces. Leaving its radial
        # clearance first is not a routed cable, even if a distant point
        # happens to lie on the opposite side of the guide plane.
        centre=np.r_[xy,0.]
        def clear_at(z):
            return clear_radius+max(0.,abs(z)-(depth/2-bevel))
        def segment_clear(start,end):
            # The authored eye widens by 0.5 mm over its final 0.5 mm.
            # Test every piecewise-linear profile boundary, not a smaller
            # fictitious straight bore. Convex radial norm attains its maximum
            # relative to linear clearance at these interval endpoints.
            samples=[start,end]
            dz=end[2]-start[2]
            if abs(dz)>1e-12:
                for plane in (-depth/2+bevel,0.,depth/2-bevel):
                    t=(plane-start[2])/dz
                    if 0<t<1:samples.append(start+t*(end-start))
            return all(np.linalg.norm(p[:2])<=clear_at(p[2]) for p in samples)
        def reaches(indices):
            previous=centre
            for j in indices:
                current=points[j]
                if abs(current[2])>=depth/2:
                    face=np.sign(current[2])*depth/2
                    t=(face-previous[2])/(current[2]-previous[2])
                    at=previous+t*(current-previous)
                    return np.sign(face) if segment_clear(previous,at) else 0
                if not segment_clear(previous,current):return 0
                previous=current
            return 0
        forward=reaches(range(i+1,len(points)))
        if reaches(range(i,-1,-1))*forward<0:
            hits.append(dict(segment=i//2,radius_m=float(np.linalg.norm(xy)),
                **({"direction":int(forward)} if signed else {})))
    return hits

def inspect(assembly,q,*,net_traversal=False,bore_clearance=False):
    """Revision 1: any crossing, centreline within bore minus cable radius.

    net_traversal: signed crossings ordered from end 0 to end 1 must not cancel,
    so a bight pushed into an eye and back out of the same face is not routed.
    bore_clearance: the centreline must cross inside the eye's hole radius;
    outside it lies the solid 11-14 mm ring, so no route outside the eye passes.
    """
    result={}
    for name,guide in assembly.routing_guides.items():
        cable=guide["cable"];spec=assembly.cable_inventory[cable]
        seg=rod_segments(assembly,q,cable)
        gap=np.linalg.norm(seg[0][1:]-seg[1][:-1],axis=1)
        max_gap=float(gap.max(initial=0.))
        clear=guide["inner_radius_m"]-(0. if bore_clearance else spec["diameter_m"]/2)
        hits=crossings(seg,guide["frame"],clear,guide["depth_m"],bevel=guide["bevel_m"],signed=net_traversal)
        through=sum(h["direction"] for h in hits)!=0 if net_traversal else bool(hits)
        result[name]=dict(cable=cable,passed_through=through and max_gap<.001,
            crossings=hits,max_rod_joint_gap_m=max_gap,
            **({"net_traversal":sum(h["direction"] for h in hits),"clear_radius_m":clear} if net_traversal else {}))
    return result

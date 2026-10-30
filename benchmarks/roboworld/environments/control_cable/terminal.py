"""Prepared ferrules and a spring-clamp terminal with an operating lever.

The terminal's exposed envelope follows WAGO 2002-1201. Internal spring/lever
dimensions are engineering estimates, not a claimed manufacturer CAD model.
"""
import math
import numpy as np
import warp as wp
import trimesh
from .geometry import newton_mesh

def add_terminal(a,name,frame):
    from shapely.geometry import box as profile_box
    b=a.builder
    inward=a.config.terminal_inward_blade
    tool_y=-.0082 if inward else .0082
    plastic=a.shape_config(ke=150000,kd=.3,mu=.30,gap=.000025,density=1200)
    def box(label,p,half,material="polycarbonate",cfg=None):
        xf=a._world(frame,p)
        return a.box(name+label,wp.transform_get_translation(xf),half,
            rotation=wp.transform_get_rotation(xf),cfg=cfg or plastic,material=material)
    # Actual wire and operating-tool apertures through the front plate.
    profile=profile_box(-.0026,-.02425,.0026,.02425)
    profile=profile.difference(profile_box(-.0017,-.0017,.0017,.0017))
    profile=profile.difference(profile_box(-.0015,tool_y-.0022,.0015,tool_y+.0023))
    front=trimesh.creation.extrude_polygon(profile,.0015,engine="earcut")
    front.apply_translation((0,0,-.0015))
    a._mesh(name+"_front",-1,front,frame,material="polycarbonate",sdf=True,cfg=plastic)
    for sign in (-1,1):
        box("_side"+str(sign),(sign*.00235,0,-.0172),(.00025,.02425,.0157))
    box("_top",(0,.0238,-.0172),(.0021,.00045,.0157))
    box("_bottom",(0,-.0238,-.0172),(.0021,.00045,.0157))
    box("_rear",(0,0,-.0324),(.0021,.02335,.0005))
    box("_din_foot",(0,-.0245,-.027),(.0026,.003,.005),material="polycarbonate")
    # Turned entry ramp guides an off-centre ferrule onto the conductor bed.
    # The former square front edge stopped insertion at only 2 mm depth.
    profile=np.array([[-.0017,-.0018,-.002],[.0017,-.0018,-.002],
        [-.0017,-.0008,-.004],[.0017,-.0008,-.004],
        [-.0017,-.0008,-.013],[.0017,-.0008,-.013],
        [-.0017,-.0020,-.013],[.0017,-.0020,-.013],
        [-.0017,-.0020,-.002],[.0017,-.0020,-.002]])
    pad=a._mesh(name+"_busbar",-1,trimesh.convex.convex_hull(profile),frame,
        material="copper",cfg=a.shape_config(ke=100000,kd=.1,mu=.40,gap=.000020,density=8800))
    a.electrodes[pad]=name+".conductor"
    # Folded rear of the conductor cage is a physical insertion end-stop.
    # Without it a force-driven fixture could push the ferrule's insulating
    # collar past the spring; the rear plastic housing is 32.9 mm deep and is
    # not the 13 mm wire-channel stop. This is estimated internal geometry.
    stop=box("_conductor_stop",(0,.0012,-.01325),(.0017,.002,.00025),
        material="copper",cfg=a.shape_config(ke=100000,kd=.1,mu=.40,
            gap=.000020,density=8800))
    a.electrodes[stop]=name+".conductor"
    # The free edge points INTO the terminal. With the old outward-facing
    # blade, axial insertion closed the clamp and withdrawal opened it.
    anchor=(0,a.config.terminal_anchor_y,-.002 if inward else -.012)
    rest=math.pi-.32 if inward else .32
    rotation=wp.quat_from_axis_angle(wp.vec3(1,0,0),rest)
    xf=a._world(frame,anchor,rotation)
    body=b.add_link(xform=xf,label=name+"_spring")
    a.cable_bodies.append(body);a.objects[name+"_spring"]=body
    E,width,thickness,length=193e9,.0028,.00034,.009
    spring_cfg=a.shape_config(ke=150000,kd=.1,mu=.45,gap=.000020,density=7900)
    lead=a.config.terminal_blade_lead_m
    if lead:
        if not 0<lead<length/3:raise ValueError("Invalid physical spring-blade lead-in")
        # An oblique physical leading face converts axial ferrule insertion
        # into opening deflection. The old square beam end has its normal
        # through the hinge and instead catches the ferrule on its outer side.
        section=[(-thickness/2,0),(thickness/2,0),
                 (thickness/2,length),(-thickness/2,length-lead)]
        vertices=[(x,y,z) for x in (-width/2,width/2) for y,z in section]
        a._mesh(name+"_blade",body,trimesh.convex.convex_hull(vertices),material="nickel",cfg=spring_cfg)
    else:
        a.box(name+"_blade",(0,0,length/2),(width/2,thickness/2,length/2),body=body,material="nickel",cfg=spring_cfg)
    # Rigid operating lever presses the same spring down, not a release flag.
    # Its pad is reached through the separate screwdriver aperture.
    # Keep the operating pad behind the front plate throughout its sweep.
    # A pad centred in the aperture at z=0 caught behind its top edge as the
    # spring deflected, mechanically latching the conductor clamp OPEN.
    desired=np.array([0,tool_y,-.004])-np.asarray(anchor)
    local=wp.quat_rotate(wp.quat_inverse(rotation),wp.vec3(*desired))
    a.box(name+"_operating_pad",local,(.0011,.0011,.0010),body=body,
        material="safety_orange",cfg=plastic)
    # Orient the rigid lever along the actual hinge-to-pad vector; the old
    # axis-aligned box left a visible and collidable gap to its operating pad.
    lever_rotation=wp.quat_from_axis_angle(wp.vec3(1,0,0),math.atan2(float(local[2]),float(local[1])))
    # The release yoke straddles the wire in X. A central bar would cross the
    # conductor's path. Pressing the negative-Y pad inward (-Z) generates the
    # positive X torque that opens the inward-facing blade.
    for x in ((-.00190,.00190) if inward else (0.,)):
        a.box(name+"_lever"+str(x),(x,float(local[1])/2,float(local[2])/2),
            (.00010 if inward else .0007,float(np.linalg.norm(local))/2,.00020 if inward else .00035),
            rotation=lever_rotation,body=body,material="nickel",cfg=spring_cfg)
    if inward:
        # The side links also clear the +/-1.7 mm busbar, not only the wire.
        # Cross-pieces close the yoke at its pivot and operating pad.
        a.box(name+"_yoke_root",(0,0,0),(.002,.00010,.00010),body=body,material="nickel",cfg=spring_cfg)
        a.box(name+"_yoke_pad_bridge",local,(.002,.00030,.00015),body=body,material="nickel",cfg=spring_cfg)
    I=width*thickness**3/12
    j=b.add_joint_revolute(-1,body,parent_xform=a._world(frame,anchor),
        axis=(1,0,0),target_pos=rest,target_ke=3*E*I/length,target_kd=.00008,
        limit_lower=math.pi-.38 if inward else -.06,
        limit_upper=math.pi+.06 if inward else .38,limit_ke=4.,limit_kd=.0001,
        label=name+"_spring_joint",collision_filter_parent=False)
    b.joint_q[b.joint_q_start[j]]=rest
    b.add_articulation([j],label=name+"_spring_mount")
    a.cable_joints.append(j);a.mechanism_joints[name+"_spring"]=j
    a._mass(body,.00045)
    a.hardware[name]=dict(kind="spring_terminal",frame=np.asarray(frame).tolist(),
        spring=body,electrodes=[pad],spring_rest_angle_rad=rest,
        spring_angular_stiffness_nm_rad=3*E*I/length,
        operating_pad_local=list(map(float,local)))
    return a.hardware[name]

def add_ferrule(a,name,frame):
    body=a._free(name,frame)
    # One mm² conductor: prepared hollow metal ferrule is already crimped,
    # so its deformation during crimping is outside this reconnecting task.
    cfg=a.shape_config(ke=100000,kd=.1,mu=.40,gap=.00002,density=8800)
    lead=a.config.ferrule_entry_lead_m
    if a.config.ferrule_square_crimp:
        if lead:raise ValueError("Square crimp and round entry-lead experiment cannot be combined")
        # Prepared square crimp, as offered by Phoenix CRIMPFOX 6S-F.
        # Equal outer cross-sectional area to the uncrimped 1.45 mm ferrule;
        # mass remains .24 g. Crimping itself is outside this task.
        half=.000725*math.sqrt(math.pi)/2
        a.box(name+"_metal",(0,0,.006),(half,half,.006),body=body,material="copper",cfg=cfg)
    elif lead:
        if not 0<lead<=.0005:raise ValueError("Ferrule lead must be in (0,0.5] mm")
        # Rolled/chamfered termination edge, an explicit estimated feature.
        # The full cross-section behind this short lead is unchanged.
        angles=np.arange(20)*2*math.pi/20
        vertices=[(r*math.cos(t),r*math.sin(t),z)
            for z,r in ((0.,.00055),(lead,.000725),(.012,.000725)) for t in angles]
        a._mesh(name+"_metal",body,trimesh.convex.convex_hull(vertices),material="copper",cfg=cfg)
    else:
        a._cylinder(name+"_metal",body,.000725,.012,(0,0,.006),material="copper",cfg=cfg)
    a.electrodes[a.builder.shape_count-1]=name+".conductor"
    a._cylinder(name+"_collar",body,.00130,.0045,(0,0,.01425),material="cable_blue",
        cfg=a.shape_config(ke=60000,kd=.2,mu=.45,gap=.00002,density=1200))
    a._mass(body,.00024)
    a.hardware[name]=dict(kind="ferrule",body=body,cable_anchor=[0,0,.0165],
        electrodes=[a.builder.shape_count-2])
    return a.hardware[name]

def add_operating_tool(a,name,frame):
    body=a._free(name,frame)
    a._cylinder(name+"_handle",body,.007,.075,(0,0,.060),material="safety_orange",
        cfg=a.shape_config(mu=.7,density=1200))
    a._cylinder(name+"_shaft",body,.0015,.028,(0,0,.013),material="nickel",
        cfg=a.shape_config(mu=.25,ke=1e6,kd=1.,density=7900))
    a.box(name+"_tip",(0,0,-.003),(.0012,.00030,.004),body=body,material="nickel",
        cfg=a.shape_config(mu=.25,ke=1e6,kd=1.,density=7900))
    a._mass(body,.022)
    return body

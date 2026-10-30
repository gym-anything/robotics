"""Product-scale workcell and native screw/robot assemblies.

M3/M4 are pre-engaged. The MuJoCo JOINT equality carries x=p*theta/(2*pi)
inside the coupled rigid solve; no policy action may set a screw coordinate.
"""

import math
from pathlib import Path
import numpy as np


def build(config, robot_setup="tighten-fr3"):
    import newton
    import warp as wp
    from newton._src.solvers.mujoco.equality import _add_equality_constraint
    from newton.solvers import SolverMuJoCo
    import robots
    from robots.backends.newton import mount

    if newton.__version__ != "1.5.0":
        raise RuntimeError("tighten_codex requires Newton 1.5.0")
    newton.use_coord_layout_targets = True
    b = newton.ModelBuilder(gravity=(0, 0, -9.81))
    SolverMuJoCo.register_custom_attributes(b)
    b.rigid_gap = config.contact_gap_m
    materials = {}
    items = {}
    cfg = lambda mu=.45, density=2700, ke=2e6, kd=150., rolling_mu=0.: newton.ModelBuilder.ShapeConfig(
        mu=mu, density=density, ke=ke, kd=kd, gap=config.contact_gap_m,
        mu_torsional=0., mu_rolling=rolling_mu)

    def box(name, pos, half, material="aluminum", body=-1, q=None, shape_cfg=None):
        i = b.add_shape_box(body, xform=wp.transform(pos, q if q is not None else wp.quat_identity()),
                            hx=half[0], hy=half[1], hz=half[2],
                            cfg=shape_cfg or cfg(), label=name)
        materials[i] = material
        return i

    def cyl(name, pos, radius, length, material="steel", body=-1, q=None, shape_cfg=None):
        i = b.add_shape_cylinder(body, xform=wp.transform(pos, q if q is not None else wp.quat_identity()),
                                 radius=radius, half_height=length/2,
                                 cfg=shape_cfg or cfg(), label=name)
        materials[i] = material
        return i

    # The table, T-slot profile and corner fixtures have the actual product scale.
    box("worktop", (0, 0, .755), (.60, .375, .025), "laminate")
    for x in (-.52, .52):
        for y in (-.31, .31):
            box("bench_leg", (x, y, .365), (.023, .023, .365), "steel")
    b.add_ground_plane(cfg=cfg(mu=.75), label="workshop_floor")
    materials[b.shape_count-1] = "concrete"
    box("back_wall", (0, 1.7, 1.55), (2, .04, 1.55), "wall")
    box("service_rail", (0, .351, 1.24), (.60, .012, .012), "safety_yellow")
    for x in (-.55, .55):
        box("service_guard_post", (x, .351, 1.01), (.012, .012, .24), "safety_yellow")
    # Flat aluminium interfaces (post, bracket, rail) are far stiffer than the
    # fastener stack; a stiff value keeps them from diluting the calibrated stacks.
    # Clamp-interface shear friction is carried by implicit friction-loss rows
    # on the sliding coordinates (env.py), with capacity mu_c x the solved
    # normal load across each interface: mujoco_warp contact friction has no
    # no-slip pass and creeps under steady shear. These faces are therefore
    # near-frictionless (mu 0.001, above MuJoCo's minimum) in the contact solver (pair mu = max of both shapes, so a
    # finger or tool touching them keeps its own mu).
    alu=cfg(mu=.001, density=0., ke=config.frame_contact_stiffness_n_m,
            kd=config.frame_contact_damping_n_s_m)
    # HFS5-2020 post, x=-10..10 mm, y=90..110 mm, top at z=1.0805 m. Its
    # front T-slot is a real 6 mm opening (two cheeks and a back web), so
    # the forward M4 shank enters the slot to the T-nut instead of passing
    # through solid stock.
    post_z=(.78+1.0805)/2
    post_h=(1.0805-.78)/2
    box("profile_post", (0, .103, post_z), (.010, .007, post_h), "aluminum", shape_cfg=alu)
    for x in (-.0065, .0065):
        box("profile_post_front_cheek", (x, .093, post_z), (.0035, .003, post_h),
            "aluminum", shape_cfg=alu)
    for x in (-.0034, .0034):
        box("post_slot_edge", (x, .0905, post_z), (.0004, .0005, post_h-.002),
            "robot_dark", shape_cfg=newton.ModelBuilder.ShapeConfig(
                has_shape_collision=False,has_particle_collision=False,density=0.))

    # Corner angle bracket (MISUMI HBLTS5-4 class, aluminium, 15 g): a free
    # body. Its vertical leg lies on the post front face and is clamped by the
    # forward M4 through a round 4.5 mm clearance hole into the post T-nut;
    # its horizontal leg rests on the post top and carries the rail, clamped
    # by the upward M4 through a 10 x 4.5 mm slot. Translation is limited
    # by the post faces (contact) and the shank in its hole (joint stops, below); rotation is fixed
    # by the leg faces and slot keys (reduced, disclosed). (Locking the
    # vertical play was tried: the rail-on-leg contact then became too
    # compliant to carry the clamp; the bracket's hopping it was meant to
    # stop was driven by the screw-carrier rattle, fixed separately.)
    bracket_origin=np.array((.013, .095, 1.075))
    bracket = b.add_link(label="corner_bracket", mass=.015,
        xform=wp.transform(bracket_origin, wp.quat_identity()),
        inertia=wp.mat33(np.diag([2.5e-6,2.5e-6,2.5e-6])))
    def bracket_box(name, xyz, half):
        return box(name, tuple(np.asarray(xyz)-bracket_origin), half, "aluminum",
                   bracket, shape_cfg=alu)
    lin=newton.ModelBuilder.JointDofConfig
    # The forward M4 shank (4 mm) in the 4.5 mm hole bounds x and z to
    # +/-0.25 mm. Those stops are joint limits, not shank contact: that
    # contact (light screw body, see the rail slot stop) stored a 200 N push
    # and threw the bracket back across its clearance at 0.17 m/s.
    hole=config.bracket_hole_clearance_m
    stop=dict(limit_ke=config.slot_stop_stiffness_n_m,limit_kd=config.slot_stop_stiffness_n_m*config.slot_stop_damping_s)
    bracket_joint=b.add_joint_d6(-1, bracket,
        linear_axes=[lin(axis=(1,0,0),limit_lower=-hole,limit_upper=hole,armature=config.bracket_armature_kg,
                         damping=config.bracket_play_damping_n_s_m,**stop),
                     lin(axis=(0,1,0),limit_lower=-.001,limit_upper=.002,armature=config.bracket_armature_kg,
                         damping=config.bracket_play_damping_n_s_m)]+
                    ([lin(axis=(0,0,1),limit_lower=-hole,limit_upper=hole,armature=config.bracket_armature_kg,
                          damping=config.bracket_play_damping_n_s_m,**stop)] if config.bracket_vertical_play else []),
        parent_xform=wp.transform(bracket_origin, wp.quat_identity()),
        label="corner_bracket_clearance_play")
    # Vertical-leg strips stop under the horizontal leg; the bend corner is
    # part of the horizontal leg, so the rail bears on one flat face only.
    for xyz,half in (((-.005625,.0875,1.065),(.003375,.0025,.0155)),
                     ((.018875,.0875,1.065),(.016625,.0025,.0155)),
                     ((0.,.0875,1.057125),(.00225,.0025,.007625)),
                     ((0.,.0875,1.074875),(.00225,.0025,.005625))):
        bracket_box("corner_bracket_vertical_leg", xyz, half)
    for xyz,half in (((.004,.10125,1.083),(.013,.01625,.0025)),
                     ((.03125,.10125,1.083),(.00425,.01625,.0025)),
                     ((.022,.091375,1.083),(.005,.006375,.0025)),
                     ((.022,.109875,1.083),(.005,.007625,.0025))):
        bracket_box("corner_bracket_horizontal_leg", xyz, half)
    # Hidden bearing lands where the M4 points bear (inside the leg faces,
    # same body): the head/washer stack stiffness is set here, separately
    # from the large flat aluminium interfaces the rail and post touch.
    land=newton.ModelBuilder.ShapeConfig(mu=.001,density=0.,
        ke=config.bearing_contact_stiffness_n_m,kd=config.top_bearing_damping_n_s_m,
        gap=config.contact_gap_m,has_shape_collision=True,is_visible=False)
    for y in (.1-.0034,.1+.0034):
        # 3 mm deep inside the 5 mm leg (bearing face unchanged at the leg's
        # underside): a 0.5 mm land let a stiff head contact tunnel through.
        box("corner_bracket_top_m4_land",tuple(np.array((.022,y,1.0820))-bracket_origin),
            (.005,.0008,.0015),"aluminum",bracket,shape_cfg=land)
    front_land=newton.ModelBuilder.ShapeConfig(mu=.001,density=0.,
        ke=config.front_bearing_contact_stiffness_n_m,kd=config.bearing_damping_n_s_m,
        gap=config.contact_gap_m,has_shape_collision=True,is_visible=False)
    for x in (-.0029,.0029):
        box("corner_bracket_front_m4_land",tuple(np.array((x,.0865,1.067))-bracket_origin),
            (.0006,.0015,.0012),"aluminum",bracket,shape_cfg=front_land)
    items["bracket"]=bracket
    assembly_joints=[bracket_joint]

    rail_origin = np.array((.14, .10, 1.0955))
    # The rail body carries the 280 mm HFS5-2020 (0.14 kg) and the SICK
    # BEF-W4-A bracket screwed to its end (2 mm stainless 1.4571: 14 mm leg
    # plus 34 x 39.4 mm front leg, ~1890 mm2 x 2 mm x 7.95 g/cm3 = 0.030 kg).
    rail_mass,bef_mass=.14,.030
    bef_com=np.array((.2665,.090,1.0867))-rail_origin
    com=bef_mass*bef_com/(rail_mass+bef_mass)
    i_rail=np.diag([9.3e-6,9.2e-4,9.2e-4])
    def shifted(I,m,d):
        d=np.asarray(d,float);return I+m*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
    rail_inertia=(shifted(i_rail,rail_mass,-com)+
                  shifted(np.diag([bef_mass*(.0394**2+.002**2)/12]*3),bef_mass,bef_com-com))
    rail = b.add_link(label="horizontal_profile", mass=rail_mass+bef_mass,
                      xform=wp.transform(rail_origin, wp.quat_identity()),
                      com=wp.vec3(*com),inertia=wp.mat33(rail_inertia))
    # The rail rests on the bracket leg by contact and is keyed to it by the
    # bracket tab in its bottom slot. It may slide along the slot (x), lift
    # (z), pitch (nose up/down) and yaw; contact with the leg, the tab and the
    # upward M4 washer - not the joint - decides where it rests. Its roll
    # about the rail axis stays locked (reduced, disclosed): the 20 mm rail
    # face lies flat across the 32.5 mm leg.
    ang=newton.ModelBuilder.JointDofConfig
    # Joint frame on the leg's far top
    # edge, the loose rail's pivot: drooping is then pure pitch, and the slot
    # friction row resists only real sliding along the slot (with the frame
    # at the rail centre, 10 mm above the contact, rotating about the edge
    # moved the slot coordinate and that row held the droop).
    rail_pivot=np.array((.0355,.10,1.0855))  # the leg's far top edge
    rail_root = b.add_joint_d6(bracket, rail,
        # Slide along the slot: the upward M4 shank (4 mm) in the bracket's
        # 10 mm slot stops it at +/-3.0 mm. The stop is this joint's limit:
        # as shank-on-slot-end contact it delivers ~0.7 MN/m, not the set
        # stiffness, because Newton converts force-space contact gains with
        # MuJoCo's direction-averaged body_invweight0 and the 2 g screw body
        # is much lighter than the rail behind it (measured). The contact sits at the leg top, at pivot height, so the
        # droop does not move it.
        linear_axes=[lin(axis=(1,0,0),limit_lower=-config.rail_slot_half_travel_m,
                         limit_upper=config.rail_slot_half_travel_m,
                         limit_ke=config.slot_stop_stiffness_n_m,
                         limit_kd=config.slot_stop_stiffness_n_m*config.slot_stop_damping_s,
                         damping=config.rail_play_damping_n_s_m),
                     # Lift at the leg edge: >= 0 is the leg's top face (a
                     # stiff one-sided stop); upward the rail is free.
                     lin(axis=(0,0,1),limit_lower=0.,limit_upper=.006,
                         limit_ke=config.rail_seat_stiffness_n_m,limit_kd=config.rail_seat_damping_n_s_m,
                         armature=config.rail_lift_armature_kg,
                         damping=config.rail_play_damping_n_s_m)],
        angular_axes=[# Pitch about the leg's far edge: >= 0 (nose down) only;
                      # nose-up would press the rail into the leg face.
                      ang(axis=(0,1,0),limit_lower=0.,limit_upper=.12,
                          limit_ke=config.rail_pitch_stop_nm_rad,
                          limit_kd=config.rail_pitch_stop_damping_nm_s_rad,
                          damping=config.rail_rotation_damping_nm_s_rad),
                      # Yaw: the bracket tab's sides in the rail's slot
                      # opening (0.2 mm per side at the tab's far corner,
                      # 32.5 mm from the pivot) as stiff stops.
                      ang(axis=(0,0,1),limit_lower=-config.rail_yaw_clearance_rad,
                          limit_upper=config.rail_yaw_clearance_rad,
                          limit_ke=config.rail_yaw_stop_nm_rad,
                          limit_kd=config.rail_yaw_stop_nm_rad*config.slot_stop_damping_s,
                          damping=config.rail_rotation_damping_nm_s_rad)],
        parent_xform=wp.transform(tuple(rail_pivot-bracket_origin),wp.quat_identity()),
        child_xform=wp.transform(tuple(rail_pivot-rail_origin),wp.quat_identity()),
        collision_filter_parent=False,label="horizontal_profile_slot_travel")
    # Bracket key tab (MISUMI HBLTS5 class): a 10 x 7.6 x 1.5 mm tab on the
    # horizontal leg entering the rail's 8 mm bottom slot opening, 0.2 mm
    # clearance per side. Its sides are the rail joint's yaw stops above; as
    # a box contact a 100 N pull at the sensor forces 2-3x the clearance and
    # then throws the rail out of its seat (screw pull, measured).
    box("corner_bracket_key_tab",tuple(np.asarray((.008,.100,1.0855+.00075))-bracket_origin),
        (.005,.0038,.00075),"aluminum",bracket,
        shape_cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False,
                                                  has_particle_collision=False,density=0.))
    if not config.frame_is_pretightened:
        # A loose corner is found drooped onto the upward M4 washer (measured
        # rest: 37.1 mrad nose down about the leg edge); the
        # construction settle confirms it rather than dropping the rail.
        q0=b.joint_q_start[rail_root]
        b.joint_q[q0+2]=.0371
    if config.task=="support":
        b.joint_q[b.joint_q_start[rail_root]]=.0015
    assembly_joints.append(rail_root)
    rail_cfg=alu
    box("horizontal_profile_top", (0, 0, .0085), (.14, .010, .0015), "aluminum", rail,
        shape_cfg=rail_cfg)
    box("horizontal_profile_web", (0, 0, 0), (.14, .004, .004), "aluminum", rail,
        shape_cfg=rail_cfg)
    for side in (-1, 1):
        box("horizontal_profile_side", (0, side*.0085, 0), (.14, .0015, .007),
            "aluminum", rail, shape_cfg=rail_cfg)
        box("horizontal_profile_slot_lip", (0, side*.007, -.0085),
            (.14, .003, .0015), "aluminum", rail, shape_cfg=rail_cfg)
    # The rail bears on the bracket leg through the rounded bottom edges of
    # its two slot lips (1 mm radius, as on the extrusion): capsules give a
    # well-defined contact normal where the rail pivots on the leg's far edge,
    # which flat box-on-box edge contact did not (a bounce there produced a
    # spurious 4 kN along the slot). The lip boxes still key on the tab.
    rail_edges=[]
    for side in ((-1, 1) if config.rail_edge_capsules else ()):
        rail_edges.append(b.add_shape_capsule(rail,
            xform=wp.transform((0., side*.007, -.009),
                               wp.quat_from_axis_angle(wp.vec3(0,1,0),math.pi/2)),
            radius=.001,half_height=.139,cfg=rail_cfg,label="horizontal_profile_lip_edge"))
        materials[rail_edges[-1]]="aluminum"
    # The rail's seat on the leg is the joint's one-sided lift and pitch
    # stops above (exact, stiff, noise-free); rail-to-leg box contact is
    # filtered. It rocked and rattled on 24 coincident soft contact points
    # under the heavy cantilever. The lips still key on the bracket tab.
    rail_ids=[i for i,l in enumerate(b.shape_label) if l.startswith("horizontal_profile_")]
    leg_ids=[i for i,l in enumerate(b.shape_label) if l=="corner_bracket_horizontal_leg"]
    for i in rail_ids:
        for j in leg_ids:
            b.add_shape_collision_filter_pair(i,j)
    items["rail"] = rail
    # Friction-loss rows on the sliding coordinates: bracket x, y, z, rail x
    # and rail yaw (lift and pitch are held by the seat stops and washer).
    # Near-hard impedance keeps static friction from creeping (with the
    # default impedance the loose rail's yaw crept 0.1 mrad/s at rest).
    bracket_dofs=[b.joint_qd_start[bracket_joint]+k for k in range(3 if config.bracket_vertical_play else 2)]
    for dof in bracket_dofs+[b.joint_qd_start[rail_root],b.joint_qd_start[rail_root]+3]:
            for key,value in (("solreffriction",(2*config.dt,1.)),
                              ("solimpfriction",(.9999,.9999,.001,.5,2.))):
                attr=b.custom_attributes["mujoco:"+key]
                if attr.values is None:attr.values={}
                attr.values[dof]=value
    # The M4 shank meeting the end of its captive slot is a hard metal stop,
    # not the softer default joint limit. Keep its measured force-space
    # stiffness/damping, but apply the same near-rigid impedance used by the
    # other metal constraints so a hard robot strike cannot cross the slot.
    slot_dof=b.joint_qd_start[rail_root]
    limit_imp=b.custom_attributes["mujoco:solimplimit"]
    if limit_imp.values is None:limit_imp.values={}
    limit_imp.values[slot_dof]=(.9999,.9999,.001,.5,2.)
    items["bracket_joint"]=bracket_joint
    items["rail_joint"]=rail_root
    def rail_visual(name, xyz, half, material, q=None):
        return box(name, tuple(np.asarray(xyz)-rail_origin), half, material, rail, q=q,
            shape_cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False,
                has_particle_collision=False,density=0.))
    # SICK BEF-W4-A (stainless 1.4571, 2 mm): the 14 mm leg is screwed to the
    # rail end; the 34 x 39.4 mm front leg carries the sensor. Its round
    # 3.4 mm hole is the pivot and its 30 deg arc slot (radius 9 mm) guides
    # the second M3, so the sensor turns in the plate plane about the pivot
    # screw. Bracket and rail are one rigid body; the plate is visual here
    # because the sensor face contact is the pivot joint below.
    bef_left,bef_right,bef_bottom,bef_top=.248,.282,1.067,1.1064
    pivot=np.array((.261,1.074)); slot_r=.009
    plate_y=(.086,.088)
    def bef_front_mesh():
        # Rectangle with a round pivot hole and a 30 deg arc slot, triangulated
        # on a fine grid for rendering only.
        xs=np.linspace(bef_left,bef_right,69);zs=np.linspace(bef_bottom,bef_top,80)
        def open_(x,z):
            dx,dz=x-pivot[0],z-pivot[1]
            r=math.hypot(dx,dz);a=math.atan2(dz,dx)
            return r<.0017 or (abs(r-slot_r)<.0017 and abs(a)<=math.radians(15)) or \
                   any(math.hypot(x-cx,z-cz)<.0017 for cx,cz in
                       ((pivot[0]+slot_r*math.cos(t),pivot[1]+slot_r*math.sin(t))
                        for t in (math.radians(-15),math.radians(15))))
        verts=[];tris=[]
        index={}
        for i,x in enumerate(xs):
            for j,z in enumerate(zs):
                for k,y in enumerate(plate_y):
                    index[i,j,k]=len(verts);verts.append((x,y,z))
        for i in range(len(xs)-1):
            for j in range(len(zs)-1):
                cx,cz=(xs[i]+xs[i+1])/2,(zs[j]+zs[j+1])/2
                if open_(cx,cz):continue
                for k in (0,1):
                    a,b_,c,d=(index[i,j,k],index[i+1,j,k],index[i+1,j+1,k],index[i,j+1,k])
                    tris.extend(((a,b_,c),(a,c,d)) if k==0 else ((a,c,b_),(a,d,c)))
        v=np.asarray(verts)-rail_origin
        return newton.Mesh(v,np.asarray(tris,dtype=np.int32).ravel(),compute_inertia=False)
    bef=b.add_shape_mesh(rail,mesh=bef_front_mesh(),
        cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False,
            has_particle_collision=False,is_visible=True,density=0.),
        label="BEF_W4_A_front_leg")
    materials[bef]="steel"
    rail_visual("BEF_W4_A_rail_end_leg",(.281,.095,(bef_bottom+bef_top)/2),
                (.001,.007,(bef_top-bef_bottom)/2),"steel")
    cyl("BEF_W4_A_fixing_head",tuple(np.array((.2835,.100,1.0955))-rail_origin),.0035,.003,"nickel",rail,
        q=wp.quat_from_axis_angle(wp.vec3(0,1,0),math.pi/2),
        shape_cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False,
            has_particle_collision=False,density=0.))

    from .kinematics import MOUNTS, ArmKinematics, cradle_layout
    setup = robots.resolve(robot_setup)
    layout = cradle_layout(setup.to_dict()["instances"])
    box("service_card", (-.10, -.20, .782), (.12, .07, .001), "paper")
    # Rubber V-groove tool holders (90 deg included angle). A round handle in
    # a V has a restoring nest, so it neither rolls nor rocks; a raised second
    # V under the shaft keeps each driver level. Rest heights follow from the
    # largest handle radius R: axis height = apex + R*sqrt(2).
    rubber=cfg(mu=.8,ke=2e6,kd=300.,rolling_mu=.02)
    tool_rest={}
    def v_saddle(name,cx,y0,y1,apex):
        hy=(y1-y0)/2;cy=(y0+y1)/2;half=.012;t=.0015
        for side in (-1,1):
            center=(cx+side*(half+t)/math.sqrt(2),cy,apex+(half-t)/math.sqrt(2))
            box(name,center,(half,hy,t),"robot_dark",
                q=wp.quat_from_axis_angle(wp.vec3(0,1,0),-side*math.pi/4),shape_cfg=rubber)
    for x,R in ((layout["M3_driver"],.018),(layout["M4_driver"],.0115)):
        apex=.797
        axis_z=apex+R*math.sqrt(2)
        box("driver_cradle_base", (x, .20, .788), (.028, .13, .008),
            "robot_dark", shape_cfg=rubber)
        v_saddle("driver_cradle_v",x,.245,.28,apex)
        v_saddle("driver_cradle_shaft_v",x,.165,.205,axis_z-.00317*math.sqrt(2))
        for side in (-1,1):
            box("driver_cradle_end_lip", (x,.20+side*.132,.801),
                (.028,.003,.005),"robot_dark",shape_cfg=rubber)
        tool_rest[x]=axis_z+.00003
    from scipy.spatial.transform import Rotation
    # Close inspection camera on a bench-clamped 20 x 20 mast with a short
    # clamp block; no long unsupported boom crosses the cell.
    box("inspection_camera_mast_foot",(.17,-.285,.786),(.035,.030,.006),"steel")
    box("inspection_camera_mast",(.17,-.285,.974),(.010,.010,.182),"aluminum")
    box("inspection_camera_clamp",(.17,-.2735,1.163),(.014,.0015,.014),"robot_dark")
    box("inspection_camera_body", (.17,-.260,1.17), (.012,.012,.011), "robot_dark")
    box("inspection_camera_lens", (.17,-.2475,1.16), (.003,.0005,.003), "red_lens")
    # The fixed station overview camera stands on the bench on a 30 x 30 mm
    # aluminium profile post bolted to a base plate, behind the camera: a
    # short bracket carries the camera body forward off the post's front
    # face (the render eye at y 0.330 m must stay outside the post).
    box("overview_camera_post_base", (0,.372,.786), (.045,.030,.006), "steel")
    box("overview_camera_post", (0,.372,1.496), (.015,.015,.704), "aluminum")
    box("overview_camera_bracket", (0,.352,2.205), (.012,.006,.010), "steel")
    overhead_q=wp.quat_from_axis_angle(wp.vec3(1,0,0),1.17)
    box("overview_camera_body", (0,.34,2.22), (.018,.019,.014),
        "robot_dark",q=overhead_q)
    box("overview_camera_lens", (0,.3322,2.2016), (.005,.003,.005),
        "red_lens",q=overhead_q)
    if config.task=="access":
        # Guard panel behind the corner and a bent lower return lip beside
        # the upward M4 corridor. Its nearest corner is 5.41 mm from the
        # top-M4 axis: 2.24 mm radial clearance for the 3.17 mm shaft.
        box("removable_access_guard", (.075,.1245,1.064),
            (.050,.002,.042), "safety_yellow")
        box("removable_access_guard_return_lip", (.075,.1065,1.059),
            (.050,.002,.009), "safety_yellow")

    def screw(name, xyz, axis, diameter, pitch, head_d, head_h, af, limit_nm,
              parent=-1, parent_origin=(0.,0.,0.), shank_length=None):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        if np.allclose(axis, [0,1,0]):
            q = wp.quat_from_axis_angle(wp.vec3(1,0,0), -math.pi/2)
        else:
            q = wp.quat_identity()
        frame = wp.transform(xyz, q)
        virtual_inertia=wp.mat33(np.diag([1e-7]*3))
        anchor = b.add_link(xform=frame, mass=.001, inertia=virtual_inertia,
                            label=name+"_nut_anchor")
        j0 = b.add_joint_fixed(parent, anchor,
            parent_xform=wp.transform(tuple(np.asarray(xyz)-np.asarray(parent_origin)), q),
            label=name+"_anchored_nut")
        guide = b.add_link(xform=frame, mass=config.screw_carrier_mass_kg, inertia=virtual_inertia,
                           label=name+"_axial_carrier")
        j1 = b.add_joint_prismatic(anchor, guide, axis=(0,0,1), target_ke=0,
            target_kd=0, damping=config.screw_axial_damping_n_s_m, limit_lower=-.0001,
            # Axial inertia reflected from the real screw's rotation through
            # its lead (the thread couples them): without it the head contact
            # sat on a 1 g carrier, and Newton's force-space conversion (by
            # body inverse weight) clamped a stiff head contact to a soft one.
            armature=(config.screw_axial_armature_m3_kg if diameter<.004 else config.screw_axial_armature_m4_kg),
            limit_ke=config.screw_axial_lock_stiffness_n_m,
            limit_kd=config.screw_axial_lock_stiffness_n_m*1e-4,
            limit_upper=.0025 if diameter<.004 else .002,
            label=name+"_pitch_advance")
        bolt = b.add_link(xform=frame, mass=.002 if diameter>.003 else .001,
                          label=name+"_screw")
        j2 = b.add_joint_revolute(guide, bolt, axis=(0,0,1), target_ke=0,
            target_kd=0, damping=.0002, friction=.005, limit_lower=-.1,
            armature=config.screw_armature_kg_m2,
            limit_ke=config.screw_lock_stiffness_nm_rad,limit_kd=.02,
            limit_upper=2*math.pi*(4 if diameter<.004 else 3),
            label=name+"_rotation")
        # Keep the small native dry-turn resistance well conditioned before
        # the head bears. Loaded thread and under-head friction are the
        # bounded elasto-plastic law in env.py.
        screw_dof=b.joint_qd_start[j2]
        for key,value in (("solreffriction",(2*config.dt,1.)),
                          ("solimpfriction",(.9999,.9999,.001,.5,2.))):
            attr=b.custom_attributes["mujoco:"+key]
            if attr.values is None:attr.values={}
            attr.values[screw_dof]=value
        if parent != -1:
            assembly_joints.extend((j0,j1,j2))
        else:
            b.add_articulation([j0,j1,j2], label=name+"_post_assembly")
        _add_equality_constraint(b, constraint_type=SolverMuJoCo.EqType.JOINT,
            joint1=j2, joint2=j1, polycoef=[0., 2*math.pi/pitch, 0., 0., 0.],
            label=name+"_thread_pitch",
            custom_attributes={"mujoco:eq_solref":(2*config.dt,1.),
                               "mujoco:eq_solimp":(.9999,.9999,.001,.5,2.)})
        metal = cfg(mu=.25, density=7850, ke=2e6, kd=100.)
        shank_length = shank_length or (.016 if diameter < .004 else .010)
        shank=cyl(name+"_shank", (0,0,shank_length/2), diameter/2, shank_length,
            "nickel", bolt, shape_cfg=metal)
        # Closed annular head with a true internal hex. Its signed-distance
        # contact supports both bit/socket and flat bearing interactions.
        ro = head_d/2
        ri = af/2
        nseg=48
        theta=np.arange(nseg)*2*math.pi/nseg
        nearest=((theta+math.pi/6)%(math.pi/3))-math.pi/6
        inner=ri/np.cos(nearest)
        points=[]
        for z in (-head_h,0.):
            for radii in (np.full(nseg,ro),inner):
                points.extend((float(r*math.cos(t)),float(r*math.sin(t)),z)
                              for r,t in zip(radii,theta))
        tris=[]
        for k in range(nseg):
            j=(k+1)%nseg
            for a0,b0,c0,d0 in ((0,nseg,2*nseg,3*nseg),):
                tris.extend(((a0+k,a0+j,c0+j),(a0+k,c0+j,c0+k),
                             (b0+k,d0+j,b0+j),(b0+k,d0+k,d0+j),
                             (c0+k,c0+j,d0+j),(c0+k,d0+j,d0+k),
                             (a0+k,b0+j,a0+j),(a0+k,b0+k,b0+j)))
        head_mesh=newton.Mesh(points,np.asarray(tris,dtype=np.int32).ravel(),
                              compute_inertia=False)
        from pathlib import Path
        head_mesh.build_sdf(device=config.device,max_resolution=192,
            narrow_band_range=(-.0005,.0005),margin=config.contact_gap_m,
            cache_dir=Path.home()/".cache/gar/tighten-sdf")
        si=b.add_shape_mesh(bolt,mesh=head_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(mu=.25,ke=2e6,kd=100.,
                gap=config.contact_gap_m,force_sdf=True,
                has_shape_collision=True,is_visible=True,density=0.),
            label=name+"_socket_head")
        materials[si]="nickel"
        # Bearing contact: exactly two small spheres over the two bearing lands
        # of the clamped part (one contact point each), on the non-rotating
        # axial carrier. A flat disk's contact
        # point count varied between ticks, and with per-point stiffness that
        # made the joint stiffness - and so the preload at a fixed angle -
        # jump by ~1.5x (538 <-> 830 N at constant screw angle). The spheres'
        # tangential friction is near zero: under-head friction is the
        # screw's friction row, and in-plane clamp friction the interface rows.
        m3=diameter<.004
        face=.0008 if name=="frame_top" else 0.
        lateral=.0026 if m3 else (.0034 if name=="frame_top" else .0029)
        offsets=((0,lateral) if name=="frame_top" else (lateral,0))
        sphere_r=.0004
        bearing=[]
        for sign in (-1,1):
            # On the axial carrier: it advances with the screw on the helix
            # but does not rotate, so the two points stay over the lands at
            # every screw angle (a rotating point would orbit over the
            # bore/slot opening and lose contact intermittently).
            bearing.append(b.add_shape_sphere(guide,
                xform=wp.transform((sign*offsets[0],sign*offsets[1],face-sphere_r),wp.quat_identity()),
                radius=sphere_r,cfg=newton.ModelBuilder.ShapeConfig(mu=.001,
                    ke=(config.m3_bearing_contact_stiffness_n_m if m3 else
                        config.front_bearing_contact_stiffness_n_m if name=="frame_front" else
                        config.bearing_contact_stiffness_n_m),
                    kd=(config.m3_bearing_damping_n_s_m if m3 else
                        config.top_bearing_damping_n_s_m if name=="frame_top" else
                        config.bearing_damping_n_s_m),
                    gap=config.contact_gap_m,has_shape_collision=True,is_visible=False,density=0.),
                label=name+"_bearing_point"))
            materials[bearing[-1]]="nickel"
        if name=="frame_top":
            # ISO 7089 M4 flat washer, 9 mm OD x 0.8 mm, bridging the slot
            # (visible; its top face carries the two bearing points).
            washer=cyl(name+"_flat_washer",(0,0,.0004),.0045,.0008,
                "nickel",bolt,shape_cfg=newton.ModelBuilder.ShapeConfig(
                    has_shape_collision=False,has_particle_collision=False,
                    is_visible=True,density=0.))
        items[name] = {"body":bolt,"slide_joint":j1,"turn_joint":j2,
                       "pitch_m":pitch,"diameter_m":diameter,"limit_nm":limit_nm,
                       "axis_world":axis.tolist(),"origin_m":list(xyz),
                       "socket_shape":si,"shank_shape":shank,
                       "bearing_shape":bearing}
        prestart=(config.fixture_hand_prestart_rad
                  if config.fixture_hand_target==name else 0.)
        if config.frame_is_pretightened and name in ("frame_top","frame_front"):
            # Secured frame: start seated (0.5 mm gap closed, zero load); the
            # construction settle tightens it (env.py) to the driver setting.
            prestart=2*math.pi*(.0005-.000002)/pitch
        if prestart:
            b.joint_q[b.joint_q_start[j1]]=pitch*prestart/(2*math.pi)
            b.joint_q[b.joint_q_start[j2]]=prestart

    # Upward M4: T-nut in the rail's bottom slot, washer 0.5 mm below the
    # bracket slot lands.
    screw("frame_top", (.022, .10, 1.0792), (0,0,1),
          .004, .0007, .007, .004, .003, config.m4_tool_limit_nm,
          parent=rail, parent_origin=rail_origin)

    # W4F (WTB4FP drawing): 16 x 37.2 x 12.1 mm, M3 holes through the 12.1 mm
    # depth at 9 mm pitch, sender axis 9 mm above the hole line. Sensor frame
    # origin is the housing center; holes at z=-14 mm.
    sensor_origin=np.array((.2655,.07995,1.088))
    # SICK weight "approx. 30 g" includes the 77 mm pigtail and M8 plug.
    # Split: 24 g housing (solid box about its centre, m/12 (b^2 + c^2)) and
    # 6 g pigtail + plug hanging below the housing (plug centre 50 mm below
    # the housing centre), which lowers the centre of mass by 10 mm.
    sensor_box=np.array((.016,.0121,.0372))
    housing_m,cable_m=.024,.006
    cable_at=np.array((0.,.003,-.050))
    sensor_com=cable_m*cable_at/(housing_m+cable_m)
    I_housing=np.diag([housing_m/12*(sensor_box[(i+1)%3]**2+sensor_box[(i+2)%3]**2) for i in range(3)])
    def _shift(I,m,d):
        d=np.asarray(d,float);return I+m*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
    sensor_inertia=_shift(I_housing,housing_m,-sensor_com)+_shift(np.zeros((3,3)),cable_m,cable_at-sensor_com)+np.eye(3)*1e-8
    sensor = b.add_link(label="WTB4FP_sensor", mass=housing_m+cable_m,
                        xform=wp.transform(sensor_origin, wp.quat_identity()),
                        com=wp.vec3(*sensor_com),inertia=wp.mat33(sensor_inertia))
    pivot_world=np.array((pivot[0],plate_y[0],pivot[1]))
    pivot_local=tuple(pivot_world-sensor_origin)
    if config.fixture_hold_sensor:
        sensor_joint=b.add_joint_fixed(rail,sensor,
            parent_xform=wp.transform(tuple(sensor_origin-rail_origin),wp.quat_identity()),
            label="sensor_TEST_ONLY_support_fixture")
    else:
        # Rotation about the pivot M3 in the plate plane; the +/-15 deg limit
        # is the arc-slot end. The joint also represents the sensor rear face
        # seated on the 2 mm plate (bilateral, disclosed).
        sensor_joint=b.add_joint_revolute(rail,sensor,
            parent_xform=wp.transform(tuple(pivot_world-rail_origin),wp.quat_identity()),
            child_xform=wp.transform(pivot_local,wp.quat_identity()),
            axis=(0,1,0),target_ke=0,target_kd=0,damping=.0005,friction=.002,
            limit_lower=-math.radians(15),limit_upper=math.radians(15),
            label="sensor_BEF_W4_A_pivot_arc_slot")
        # A loose W4F swings under its own weight to the slot end; the scene
        # starts at that measured rest rather than dropping at tick 0.
        b.joint_q[b.joint_q_start[sensor_joint]]=math.radians(15)
    if not config.fixture_hold_sensor:
        # Near-hard impedance for the pivot's clamp friction row (see env.py);
        # the default soft row creeps freely on a 30 g sensor.
        dof=b.joint_qd_start[sensor_joint]
        for key,value in (("solreffriction",(2*config.dt,1.)),
                          ("solimpfriction",(.9999,.9999,.001,.5,2.))):
            attr=b.custom_attributes["mujoco:"+key]
            if attr.values is None:attr.values={}
            attr.values[dof]=value
    assembly_joints.append(sensor_joint)
    items["sensor_aim_joint"]=sensor_joint
    items["sensor_pivot_world_m"]=pivot_world.tolist()
    nocol=lambda:newton.ModelBuilder.ShapeConfig(has_shape_collision=False,
        has_particle_collision=False,is_visible=True,density=0.)
    plastic = cfg(mu=.48, density=0., ke=1e6, kd=150.)
    box("sensor_housing", (0, 0, .0046), (.008, .00605, .014), "sensor_blue", sensor,
        shape_cfg=plastic)
    box("sensor_rounded_top", (0,0,.0189), (.0072,.0055,.0003), "sensor_blue", sensor,
        shape_cfg=nocol())
    box("sensor_front_window", (0,-.0062,.004), (.0062,.0003,.0115), "screen",
        sensor, shape_cfg=nocol())
    for z,mat in ((-.005,"red_lens"),(.003,"robot_dark")):
        cyl("sensor_optic_"+mat, (0,-.0064,z), .0024, .0006, mat, sensor,
            q=wp.quat_from_axis_angle(wp.vec3(1,0,0),math.pi/2), shape_cfg=nocol())
    box("sensor_display_keys", (0,.001,.0188), (.004,.003,.0006), "robot_dark", sensor,
        shape_cfg=nocol())
    box("sensor_status_led", (.005,-.004,.0188), (.0012,.0012,.0006), "status_green",
        sensor, shape_cfg=nocol())
    items["sensor_optic_local_m"]=(0.,-.0065,-.005)
    for dx in (-.0045, .0045):
        # Round-hole lobe: 3.4 mm bore, 7.0 mm outside diameter (visual), and
        # two hidden bearing lands spanning radius 1.7-3.5 mm, the bore edge
        # to the lobe edge, which carry the head load.
        segments=32
        vertices=[]
        for y,radius in ((-.00605,.0035),(-.00605,.0017),
                         (.00605,.0035),(.00605,.0017)):
            vertices.extend((dx+radius*math.cos(2*math.pi*k/segments),y,
                             -.014+radius*math.sin(2*math.pi*k/segments))
                            for k in range(segments))
        triangles=[]
        for k in range(segments):
            j=(k+1)%segments
            for a0,b0,c0,d0 in ((0,segments,2*segments,3*segments),):
                triangles.extend(((a0+k,a0+j,b0+j),(a0+k,b0+j,b0+k),
                                  (c0+k,d0+j,c0+j),(c0+k,d0+k,d0+j),
                                  (a0+k,c0+j,a0+j),(a0+k,c0+k,c0+j),
                                  (b0+k,b0+j,d0+j),(b0+k,d0+j,d0+k)))
        ear_mesh=newton.Mesh(vertices,np.asarray(triangles,dtype=np.int32).ravel(),
                             compute_inertia=False)
        visual=b.add_shape_mesh(sensor,mesh=ear_mesh,cfg=nocol(),
            label="sensor_M3_round_ear_visual")
        materials[visual]="sensor_blue"
        for side in (-1, 1):
            box("sensor_mount_ear", (dx+side*.0026, 0, -.014),
                (.0009, .00605, .0030), "sensor_blue", sensor,
                shape_cfg=newton.ModelBuilder.ShapeConfig(mu=.001,density=0.,
                    ke=config.m3_bearing_contact_stiffness_n_m,
                    kd=config.m3_bearing_damping_n_s_m,gap=config.contact_gap_m,
                    has_shape_collision=True,is_visible=False))
    # Threaded M3 clamping strip behind the plate (moves with the sensor).
    box("sensor_M3_clamping_strip",(0,.00805+.001,-.014),(.0085,.001,.003),"steel",
        sensor,shape_cfg=nocol())
    # The pigtail leaves the housing bottom and hangs to its M8 plug.
    lead=((0.,.001,-.0186),(0.,.003,-.030),(0.,.003,-.038),(0.,.003,-.044))
    for k,(a,c) in enumerate(zip(lead,lead[1:])):
        direction=np.asarray(c)-np.asarray(a)
        length=float(np.linalg.norm(direction))
        quat=Rotation.align_vectors([direction/length],[[0,0,1]])[0].as_quat()
        cyl(f"sensor_pigtail_{k}",tuple((np.asarray(a)+np.asarray(c))/2),
            .0017,length,"cable_black",sensor,q=wp.quat(*quat),shape_cfg=nocol())
    cyl("sensor_M8_cable_end",(0.,.003,-.050),.004,.012,"cable_black",sensor,shape_cfg=nocol())
    items["sensor"] = sensor
    # Both M3s thread into the clamping strip, so their thread and head
    # reactions act on the sensor itself (not on a world or rail anchor).
    for side, dx in (("left",-.0045),("right",.0045)):
        screw("sensor_"+side, tuple(sensor_origin+np.array((dx,-.00605-.0006,-.014))),
              (0,1,0), .003, .0005, .0055, .003, .0025, config.m3_tool_limit_nm,
              parent=sensor, parent_origin=sensor_origin)
    b.add_articulation(assembly_joints, label="captured_rail_sensor_and_fasteners")

    # The white reference flag slides aside for the screwdriver handle, then
    # returns for the optical check.
    flag_x=float(sensor_origin[0])
    box("flag_slider_track",(flag_x+.05,-.063,.779),(.065,.008,.001),"steel")
    flag_origin=(flag_x,-.063,.934)
    flag=b.add_link(xform=wp.transform(flag_origin,wp.quat_identity()),
                    mass=.050,label="reference_flag_carriage")
    flag_joint=b.add_joint_prismatic(-1,flag,
        parent_xform=wp.transform(flag_origin,wp.quat_identity()),
        axis=(1,0,0),target_ke=0,target_kd=0,damping=.8,friction=.03,
        limit_lower=-.006,limit_upper=.10,label="reference_flag_retract")
    # A guided M3 test driver staged on the socket axis passes through the
    # flag plane; without its parking stage the test starts with the flag
    # aside (disclosed). All ordinary scenes start with the flag in place.
    guided_m3=(config.fixture_mode and not config.fixture_no_driver and
               config.fixture_hand_driver is None and
               (config.fixture_driver or ("M3_driver" if config.fixture_target.startswith("sensor_")
                                          else "M4_driver"))=="M3_driver")
    parked=guided_m3 and config.fixture_dual_driver
    b.joint_q[b.joint_q_start[flag_joint]]=(.10 if guided_m3 and not parked else
                                            -.0035 if config.task=="aim" else 0.)
    box("white_alignment_flag",(0,.005,.149),(.005,.001,.022),"paper",flag)
    box("flag_stand",(0,0,0),(.004,.004,.1535),"steel",flag)
    box("flag_foot",(0,0,-.149),(.020,.025,.005),"steel",flag)
    b.add_articulation([flag_joint],label="sliding_reference_flag")
    items["flag"]=flag
    items["flag_joint"]=flag_joint
    items["flag_center_z_m"]=.934+.149
    items["flag_face_y_m"]=-.063+.006

    # Forward M4 threads into the post's front T-nut (world-fixed nut; the
    # bracket's 4.5 mm hole gives the loose joint its 0.25 mm play).
    screw("frame_front", (0., .0845, 1.067), (0,1,0),
          .004, .0007, .007, .004, .003, config.m4_tool_limit_nm, parent=-1)

    def driver(name, x, limit_nm, af, engaged=False, hand_staged=False,
               fixture_target=None):
        if hand_staged:
            # Assisted initial placement of the handle between the *actual*
            # selected FR3 fingers. The tool is otherwise a free body, not welded
            # to the robot. Its local shaft points along world +x so the grip
            # segment crosses both opposing finger pads.
            if config.fixture_hand_target is not None:
                target=items[config.fixture_hand_target]
                stand_off=(.040 if config.fixture_hand_target=="sensor_left" and
                           config.fixture_hand_prestart_rad>10 else .041)
                tool_pos=tuple(np.asarray(target["origin_m"])-
                               np.asarray(target["axis_world"])*stand_off)
                base=(wp.quat_from_axis_angle(wp.vec3(1,0,0),math.pi)
                      if config.fixture_hand_target=="frame_top" else
                      wp.quat_from_axis_angle(wp.vec3(1,0,0),math.pi/2))
                bit_phase=config.fixture_hand_prestart_rad
                if config.fixture_hand_target=="sensor_left" and bit_phase>10:
                    # A hex bit has six equivalent phases. Choose the one
                    # nearest the reachable right-hand staging posture.
                    bit_phase=8.2+((bit_phase-8.2+math.pi/6)%(math.pi/3)-math.pi/6)
                tool_rot=wp.mul(wp.quat_from_axis_angle(
                    wp.vec3(*target["axis_world"]),bit_phase),base)
            else:
                tool_pos=(-.0455,-.26,1.233)
                tool_rot=wp.quat_from_axis_angle(wp.vec3(0,1,0),math.pi/2)
        elif engaged:
            target=items[fixture_target or config.fixture_target]
            axis=np.asarray(target["axis_world"])
            origin=np.asarray(target["origin_m"])
            wrong_bit=(name.startswith("M3") !=
                       (fixture_target or config.fixture_target).startswith("sensor_"))
            tool_pos=tuple(origin-axis*(.045 if wrong_bit else .041))
            tool_rot=wp.quat_from_axis_angle(wp.vec3(1,0,0),
                                             math.pi if axis[2]>.5 else math.pi/2)
        else:
            tool_pos=(x,.20,tool_rest[x])
            tool_rot=wp.quat_from_axis_angle(wp.vec3(1,0,0),-math.pi/2)
        tool_frame = wp.transform(tool_pos, tool_rot)
        slim=name.startswith("M4")
        # Product masses: Wera 7440 180 g, HansTool 0176 32 g + 6 g collar.
        # The steel shaft/bit rotor is ~10.5 g from its own shapes; handle
        # densities give the remainder, so inertia follows the geometry.
        grip = b.add_link(xform=tool_frame, mass=0., label=name+"_grip")
        if engaged:
            transfer_joint=None
            transfer_carrier=None
            # Lateral and vertical TEST_ONLY stages for the M3 guide: the
            # fixture is built on the design screw axis, but a pre-tightened
            # frame settles the sensor ~0.1-0.2 mm away, enough to land a
            # 2.5 mm hex bit on the socket rim. Assisted cases centre the bit
            # on the socket's settled position with these stages.
            if (fixture_target or config.fixture_target)=="sensor_left" and config.fixture_withdraw_driver:
                transfer_carrier=b.add_link(xform=tool_frame,mass=.001,
                    inertia=wp.mat33(np.diag([1e-7]*3)),
                    label=name+"_TEST_ONLY_lateral_carrier")
                transfer_joint=b.add_joint_prismatic(-1,transfer_carrier,
                    parent_xform=tool_frame,axis=(1,0,0),target_ke=6000.,
                    target_kd=80.,damping=2.,limit_lower=-.060,limit_upper=.012,
                    label=name+"_TEST_ONLY_lateral_stage")
                # Dual: staged parked 50 mm aside, clear of the reference flag.
                if config.fixture_dual_driver:
                    b.joint_q[b.joint_q_start[transfer_joint]]=-.050
                # A vertical test stage follows the arc-slot screw, which rises
                # or falls as the sensor turns about its pivot M3.
                lift_carrier=b.add_link(xform=tool_frame,mass=.001,
                    inertia=wp.mat33(np.diag([1e-7]*3)),
                    label=name+"_TEST_ONLY_vertical_carrier")
                lift_joint=b.add_joint_prismatic(transfer_carrier,lift_carrier,
                    axis=(0,1,0),target_ke=6000.,target_kd=80.,damping=2.,
                    limit_lower=-.004,limit_upper=.004,
                    label=name+"_TEST_ONLY_vertical_stage")
            if config.fixture_withdraw_driver:
                # A fixture prismatic stage lets an assisted experiment
                # withdraw the real hex bit after preload has settled. The
                # normal robot environment still has a free driver body.
                carrier=b.add_link(xform=tool_frame,mass=.001,
                    inertia=wp.mat33(np.diag([1e-7]*3)),
                    label=name+"_TEST_ONLY_retraction_carrier")
                slide=b.add_joint_prismatic(lift_carrier if transfer_carrier is not None else -1,
                    carrier,parent_xform=tool_frame if transfer_carrier is None else wp.transform_identity(),
                    axis=(0,0,1),target_ke=6000.,target_kd=80.,damping=2.,
                    limit_lower=-.002,limit_upper=.025,
                    label=name+"_TEST_ONLY_retraction_stage")
                if config.fixture_approach_driver:
                    b.joint_q[b.joint_q_start[slide]]=.020
                root=b.add_joint_revolute(carrier,grip,axis=(0,0,1),
                    target_ke=0,target_kd=0,damping=.001,friction=.001,
                    limit_lower=-100,limit_upper=100,
                    label=name+"_TEST_ONLY_rotary_guide")
            else:
                slide=None
                root=b.add_joint_revolute(-1,grip,parent_xform=tool_frame,
                    axis=(0,0,1),target_ke=0,target_kd=0,damping=.001,
                    friction=.001,limit_lower=-100,limit_upper=100,
                    label=name+"_TEST_ONLY_rotary_guide")
        else:
            slide=None
            transfer_joint=None
            root=b.add_joint_free(child=grip,label=name+"_free")
        handle_kd=config.m4_handle_damping_n_s_m if slim else config.m3_handle_damping_n_s_m
        handle_cfg = cfg(mu=.75,density=1457. if slim else 1926.,ke=3e5,kd=handle_kd)
        segments=(((.022,.030,.0063),(.052,.055,.0069),(.087,.015,.0063))
                  if slim else
                  ((.030,.035,.011),(.065,.055,.018),(.105,.020,.014)))
        for k,(z,length,r) in enumerate(segments):
            cyl(name+f"_handle_{k}", (0,0,z), r, length,
                "tool_blue" if k==1 else "robot_dark", grip, shape_cfg=handle_cfg)
        if not slim:
            # Wera 7440: hexagonal anti-roll section at the handle end
            # (visual; contact uses the round section inside it).
            hx=[]
            for zz in (.098,.112):
                hx+=[(.0165*math.cos(a),.0165*math.sin(a),zz) for a in np.arange(6)*math.pi/3]
            hf=[]
            for k in range(6):
                j=(k+1)%6
                hf+=[(k,j,6+j),(k,6+j,6+k)]
            hf+=[(0,2,1),(0,3,2),(0,4,3),(0,5,4),(6,7,8),(6,8,9),(6,9,10),(6,10,11)]
            hex_shape=b.add_shape_mesh(grip,mesh=newton.Mesh(np.asarray(hx),np.asarray(hf,np.int32).ravel(),
                compute_inertia=False),cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False,
                has_particle_collision=False,is_visible=True,density=0.),label=name+"_anti_roll_hex")
            materials[hex_shape]="robot_dark"
        collar_shape=None
        if slim:
            # Authored 23 mm OD, 30 mm long hollow gripper collar on the
            # manufacturer's 13.8 mm handle. The extra 6 g is in grip mass.
            # Its circular section can rotate beside the post without an
            # eccentric corner sweeping into the extrusion.
            collar_shape=cyl(name+"_robot_grip_collar",(0,0,.065),.0115,.030,
                "robot_dark",grip,shape_cfg=cfg(mu=.9,density=482.,ke=3e5,kd=handle_kd))
        cyl(name+"_rating_band", (0,0,.072 if slim else .087),
            .007 if slim else .0185, .006,
            "tool_green" if limit_nm<.5 else "tool_yellow", grip, shape_cfg=handle_cfg)
        rotor = b.add_link(xform=tool_frame, mass=0., label=name+"_shaft_and_bit")
        # One-way detent clutch (env.py drives it each tick): the relative
        # handle-bit coordinate is held by a zero-width limit (locked, both
        # ways) until the transmitted tightening torque reaches the setting;
        # the cam then lets it travel one notch against a reduced roll-over
        # torque and relocks one notch further. Loosening never slips.
        j = b.add_joint_revolute(grip,rotor,axis=(0,0,1),target_ke=0,target_kd=0,
            friction=0.,limit_lower=-1.,limit_upper=1.,
            limit_ke=config.clutch_lock_stiffness_nm_rad,
            limit_kd=config.clutch_lock_damping_nm_s_rad,
            armature=config.clutch_armature_kg_m2,
            collision_filter_parent=True,label=name+"_detent_clutch")
        b.add_articulation(([transfer_joint,lift_joint] if transfer_joint is not None else [])+
                           ([slide] if slide is not None else [])+[root,j],
                           label=name+"_torque_driver")
        steel_cfg = cfg(mu=.35,density=7850,ke=1e6,kd=150.)
        cyl(name+"_steel_shaft", (0,0,-.011), .00317,.040,
            "nickel",rotor,shape_cfg=steel_cfg)
        # A true external hex with a 0.5 mm lead-in chamfer. Longer tapers
        # left too little full-flat engagement in the 3 mm-deep M3 socket as
        # the screw advanced under the test guide.
        # Drive flats 50 um below nominal AF: hex bit/socket fit clearance on
        # the order of ISO 1173 bit and ISO 4762 socket tolerances (100 um
        # gives excess backlash).
        bit_af = af-.00005
        radius = bit_af/(2*math.cos(math.pi/6))
        az = math.pi/6+np.arange(6)*math.pi/3
        verts = np.array([(scale*radius*math.cos(t),scale*radius*math.sin(t),z)
                          for z,scale in ((-.040,.80),(-.0395,1.),(-.022,1.))
                          for t in az],dtype=float)
        faces=[]
        for ring in (0,1):
            for k in range(6):
                k2=(k+1)%6
                lo=6*ring;hi=6*(ring+1)
                faces.extend(((lo+k,lo+k2,hi+k2),(lo+k,hi+k2,hi+k)))
        faces.extend(((0,2,1),(0,3,2),(0,4,3),(0,5,4)))
        faces.extend(((12,13,14),(12,14,15),(12,15,16),(12,16,17)))
        mesh=newton.Mesh(vertices=verts,indices=np.asarray(faces,dtype=np.int32).ravel())
        si=b.add_shape_mesh(rotor,mesh=mesh,cfg=steel_cfg,label=name+"_hex_bit")
        materials[si]="nickel"
        # Friction loss is passive and dissipative. Near-rigid impedance stops
        # a light rotor creeping through the nominal setting under small loads.
        dof=b.joint_qd_start[j]
        for key,value in (("solreffriction",(2*config.dt,1.)),
                          ("solimpfriction",(.9999,.9999,.001,.5,2.))):
            attr=b.custom_attributes["mujoco:"+key]
            if attr.values is None: attr.values={}
            attr.values[dof]=value
        items[name]={"grip":grip,"rotor":rotor,"clutch_joint":j,
                     "guide_root_joint":root,
                     "guide_slide_joint":slide,
                     "guide_transfer_joint":transfer_joint,
                     "guide_lift_joint":lift_joint if transfer_joint is not None else None,
                     "guide_stage_axes_world":(np.asarray(wp.quat_rotate(tool_rot,wp.vec3(1,0,0))).tolist(),
                                               np.asarray(wp.quat_rotate(tool_rot,wp.vec3(0,1,0))).tolist()),
                     "bit_shape":si,
                     "grip_collar_shape":collar_shape,
                     "limit_nm":limit_nm,"bit_af_m":bit_af,
                     "mass_kg":.038 if slim else .180}

    chosen=(config.fixture_driver or
            ("M3_driver" if config.fixture_target.startswith("sensor_") else "M4_driver"))
    driver("M3_driver",layout["M3_driver"],config.m3_tool_limit_nm,.0025,
           config.fixture_mode and not config.fixture_no_driver and
           config.fixture_hand_driver is None and chosen=="M3_driver",
           config.fixture_hand_driver=="M3_driver")
    driver("M4_driver",layout["M4_driver"],config.m4_tool_limit_nm,.003,
           config.fixture_mode and not config.fixture_no_driver and
           config.fixture_hand_driver is None and chosen=="M4_driver",
           config.fixture_hand_driver=="M4_driver")
    # A second guide exists only in coupled diagnostic scenes. Each turns a
    # separate real socket; neither constrains the normal rail/sensor joint.
    if config.fixture_dual_driver:
        if config.fixture_target=="frame_top":
            driver("M4_driver_front",-.14,config.m4_tool_limit_nm,.003,
                   engaged=True,fixture_target="frame_front")

    # The SDF socket and primitive bearing are two views of the same head.
    # Against each intended clamped face, only the bearing primitive collides.
    for name,item in items.items():
        if not isinstance(item,dict) or "socket_shape" not in item:continue
        labels=({"sensor_mount_ear"} if name.startswith("sensor_") else
                {"corner_bracket_horizontal_leg","corner_bracket_top_m4_land"} if name=="frame_top" else
                {"corner_bracket_vertical_leg","corner_bracket_front_m4_land"})
        for shape_id,label in enumerate(b.shape_label):
            if label in labels:
                b.add_shape_collision_filter_pair(item["socket_shape"],shape_id)
        if name in ("frame_top","frame_front"):
            # The shanks' stops in the bracket slot/hole are joint limits.
            leg_labels=({"corner_bracket_horizontal_leg"} if name=="frame_top" else
                        {"corner_bracket_vertical_leg","corner_bracket_front_m4_land"})
            for shape_id,label in enumerate(b.shape_label):
                if label in leg_labels:
                    b.add_shape_collision_filter_pair(item["shank_shape"],shape_id)
        if not name.startswith("sensor_"):
            # The bearing points touch only their dedicated lands.
            leg=("corner_bracket_horizontal_leg" if name=="frame_top" else
                 "corner_bracket_vertical_leg")
            for shape_id,label in enumerate(b.shape_label):
                if label==leg:
                    for point in item["bearing_shape"]:
                        b.add_shape_collision_filter_pair(point,shape_id)

    # Catalog assembly owns URDF/controller parameters. The world owns mounts,
    # solver, task and clock. The default rig is explicitly selected by GAR.
    pending = robots.add_to(b, setup, mounts={
        name: mount(position, yaw) for name,(position,yaw) in MOUNTS.items()})
    # Clamp interfaces (post, bracket, rail, M3 ears, head/washer faces) get
    # a near-hard contact impedance so static friction does not creep.
    if config.clamp_contact_dmax>.95:
        attr=b.custom_attributes["mujoco:geom_solimp"]
        if attr.values is None:attr.values={}
        clamp_labels=("profile_post","profile_post_front_cheek","corner_bracket_vertical_leg",
                      "corner_bracket_horizontal_leg","corner_bracket_key_tab","horizontal_profile_lip_edge","horizontal_profile_top","horizontal_profile_web",
                      "horizontal_profile_side","horizontal_profile_slot_lip","sensor_mount_ear")
        for i,label in enumerate(b.shape_label):
            if label in clamp_labels or label.endswith("_bearing_point"):
                attr.values[i]=(.9,config.clamp_contact_dmax,.001,.5,2.)
    # Shapes welded into one rigid body (e.g. the Franka Hand collapsed onto
    # FR3 link 7) can only exert internal, cancelling contact forces. Filter
    # those pairs: they carry no physics but occupy solver rows and history.
    by_body={}
    for i,body in enumerate(b.shape_body):
        if body>=0:by_body.setdefault(body,[]).append(i)
    for shapes in by_body.values():
        for k,i in enumerate(shapes):
            for j in shapes[k+1:]:
                b.add_shape_collision_filter_pair(i,j)
    # FR3 self-collision: filter exactly the manufacturer's SRDF
    # disable_collisions pairs within each arm (Adjacent / Never / Default).
    # They are 720 of ~840 resting contact
    # candidates, all force-free, and cost collision and history time.
    import xml.etree.ElementTree as ET
    srdf=ET.parse(Path(__file__).with_name("robot_collision")/"fr3_franka_hand.srdf").getroot()
    merged={"fr3_hand":"fr3_link7","fr3_link8":"fr3_link7"}
    disabled={frozenset((merged.get(e.get("link1"),e.get("link1")),merged.get(e.get("link2"),e.get("link2"))))
              for e in srdf.iter("disable_collisions")}
    link_shapes={}
    for i,body in enumerate(b.shape_body):
        if body>=0:
            parts=b.body_label[body].split("/")
            if len(parts)==3 and parts[1]=="fr3" and parts[0] in ("left","right"):
                link_shapes.setdefault((parts[0],parts[2]),[]).append(i)
    for (arm,la),sa in link_shapes.items():
        for (arm2,lb),sb in link_shapes.items():
            if arm2!=arm or la>=lb or frozenset((la,lb)) not in disabled:
                continue
            for i in sa:
                for j in sb:
                    b.add_shape_collision_filter_pair(i,j)
    if config.fixture_hand_target is not None:
        # Assisted initial robot/tool placement, solved from the catalog URDF.
        # Subsequent motion is normal FR3 actuation and physical tool contact.
        from scipy.spatial.transform import Rotation as R
        arm=config.fixture_hand_arm
        kin=ArmKinematics(pending,setup)
        indexes=[b.joint_q_start[b.joint_label.index(
            pending.joints[name]["native_label"])] for name in kin.names[arm]]
        seed=[float(b.joint_q[i]) for i in indexes]
        target=items[config.fixture_hand_target]
        axis=np.asarray(target["axis_world"])
        yaw_angle=(-1.3 if config.fixture_hand_target=="frame_top" else
                   (0. if arm=="right" else -1.8))
        yaw=R.from_rotvec(yaw_angle*axis)
        # Point the robot wrist away from the workpiece along the handle.
        base=R.from_euler("x",0. if config.fixture_hand_target=="frame_top" else -math.pi/2)
        orientation=(yaw*base).as_quat()
        tool_origin=np.asarray(target["origin_m"])-axis*(
            .047 if config.fixture_hand_target=="sensor_left" else .041)
        collar=tool_origin+(yaw*base).apply((0.,0.,.065))
        # Measured default-hand TCP-to-grasp-center offset, transformed with
        # the new wrist orientation. The IK source is the exact catalog URDF.
        default_tcp=kin.fk(arm,seed)
        default_rot=R.from_matrix(default_tcp[:3,:3])
        default_collar=np.asarray((.0195 if arm=="left" else .8795,-.26,1.233))
        local_offset=default_rot.inv().apply(default_collar-default_tcp[:3,3])
        tcp=collar-(yaw*base).apply(local_offset)
        if config.fixture_hand_target=="sensor_left" and arm=="right":
            # The hand's collision fingers extend 90 mm forward of the
            # catalog TCP in this rotated placement. Seat their actual pad
            # centers on the handle, leaving the wrist behind the tool.
            tcp+=np.asarray((0.,-.090,0.))
        fit=kin.solve(arm,tcp,orientation,seed)
        if not fit["converged"]:
            raise RuntimeError("Assisted hand starting pose is unreachable")
        for index,value in zip(indexes,fit["joints"]):b.joint_q[index]=value
    if config.task=="mixed":
        # Task prop (added last so every existing shape keeps its index): a
        # steel home stop with a damped bumper face on the worktop at the
        # track's return end. Sliding the flag back until its foot meets the
        # stop leaves it 0.1 mm past its home, where the sensor sees it;
        # without it the only return-side stop was the slide's -6 mm limit,
        # where the beam misses. The flag's own joint, limits, damping and
        # friction are unchanged. Contact gains of this prop only: Newton's
        # force-space mapping scales the pair's mean ke/kd by the 0.16 kg
        # carriage's body_invweight0 x (1 - dmax) = 2.09 x 0.05; with the
        # default 2e6/150 the pair had a 128 ms time constant and damping
        # ratio 0.017 and threw the flag back out of the beam (measured).
        # These give 4 dt and a damping ratio of ~1.
        box("flag_home_stop",(flag_x-.020-.0001-.004,-.063,.785),(.004,.010,.005),"robot_dark",
            shape_cfg=cfg(mu=.45,ke=2.5e7,kd=4.6e4))
    for i in range(b.shape_count):
        if i not in materials:
            materials[i] = "robot_white"
    return b, items, materials, pending, setup

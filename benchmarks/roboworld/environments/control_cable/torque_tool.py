"""Product-scale 0.4 Nm driver with a passive native friction clutch.

Envelope/masses: Phoenix 1208429 + 1208432. Internal mass split, grip contour,
jaw dimensions and friction-clutch approximation are declared estimates.
No tool-to-connector attachment, automatic alignment or nut drive is installed.
The D15 jaw's loaded engagement remains a separate physical qualification gate.
"""
import math
import numpy as np
import warp as wp
import trimesh
from shapely.geometry import Point,box

def add_driver(a,name,frame,*,free_handle=True,handle_support=None):
    b=a.builder
    first=b.joint_count
    # One ordered articulation, not add_body's independent free articulation
    # followed by a second child articulation. eval_fk runs articulations in
    # parallel; a cross-articulation dynamic parent gives stale/racing twists.
    handle=b.add_link(xform=frame,label=name+"_handle")
    if free_handle:
        if handle_support is not None:raise ValueError("Free handle cannot also have a support")
        root=b.add_joint_free(child=handle,label=name+"_free_joint")
    else:
        if handle_support is None:raise ValueError("Supported fixture requires an explicit root joint")
        root=handle_support(b,handle)
    a.rigid_bodies.append(handle);a.rigid_joints.extend(range(first,b.joint_count))
    a.objects[name+"_handle"]=handle
    # Dissipative numerical contact prior near critical point-contact damping;
    # unforced six-second table test is recorded, not manufacturer calibration.
    plastic=a.shape_config(ke=100000,kd=a.config.driver_contact_damping_ns_m,mu=.75,density=1200)
    # 105 mm handle + 100 mm exposed blade = 205 mm overall driver length.
    # Piecewise convex grip sections keep collision and visible contours equal.
    levels=[(0,.009),(.014,.013),(.030,.0175),(.060,.0185),(.090,.0165),(.105,.012)]
    angles=np.arange(12)*2*math.pi/12
    for j,((z0,r0),(z1,r1)) in enumerate(zip(levels,levels[1:])):
        vertices=[(r*math.cos(t),r*math.sin(t),z) for z,r in ((z0,r0),(z1,r1)) for t in angles]
        a._mesh(name+"_grip_section_"+str(j),handle,trimesh.convex.convex_hull(vertices),
            material="cable_black",cfg=plastic)
    for z in (.023,.081):
        a._cylinder(name+"_grip_band",handle,.0177,.004,(0,0,z),material="paint_blue",cfg=plastic)
    a._mass(handle,.0799)
    rotor=b.add_link(xform=frame,label=name+"_shaft_and_bit")
    a.rigid_bodies.append(rotor);a.objects[name+"_shaft_and_bit"]=rotor
    steel=a.shape_config(ke=1e6,kd=a.config.driver_contact_damping_ns_m,mu=.3,gap=.000025,density=7900)
    # 4 mm across flats, external hexagonal drive, not a thick round shaft.
    radius=.002/math.cos(math.pi/6)
    vertices=[(radius*math.cos(t),radius*math.sin(t),z)
              for z in (-.100,.004) for t in np.arange(6)*math.pi/3]
    a._mesh(name+"_hex_blade",rotor,trimesh.convex.convex_hull(vertices),material="nickel",cfg=steel)
    # An estimated side-entry crescent accessory. Its offset drive and real
    # open jaw follow the inspected product family photo, not a closed socket.
    centre=(-.017,0)
    profile=Point(*centre).buffer(.014,quad_segs=32).difference(Point(*centre).buffer(.0075,quad_segs=32))
    profile=profile.difference(box(-.05,-.0076,-.017,.0076))
    profile=profile.union(box(-.006,-.0035,.003,.0035))
    jaw=trimesh.creation.extrude_polygon(profile,.006,engine="earcut")
    jaw.apply_translation((0,0,-.100))
    a._mesh(name+"_D15_side_entry_bit",rotor,jaw,material="nickel",sdf=True,cfg=steel)
    a._mass(rotor,.0290)  # 11.0 g driver blade + 18.0 g bit; total 108.9 g.
    clutch=b.add_joint_revolute(handle,rotor,axis=(0,0,1),
        target_ke=0,target_kd=0,friction=.4,limit_lower=-1e6,limit_upper=1e6,
        collision_filter_parent=True,label=name+"_passive_torque_clutch")
    b.add_articulation([root,clutch],label=name+"_manufactured_driver")
    a.rigid_joints.append(clutch)
    # MuJoCo friction-loss rows have velocity residual, not position residual.
    # Default .9 impedance / .02 s decay allowed tens of rad/s creep on this
    # small-inertia driver under just .2 Nm. Use explicit near-rigid impedance
    # and the documented safe 2*dt decay, without increasing the .4 Nm cap.
    # https://mujoco.readthedocs.io/en/latest/modeling.html#friction
    dof=b.joint_qd_start[clutch]
    for key,value in (("solreffriction",(2*a.config.dt,1.)),
                      ("solimpfriction",(.9999,.9999,.001,.5,2.))):
        attribute=b.custom_attributes["mujoco:"+key]
        if attribute.values is None:attribute.values={}
        attribute.values[dof]=value

    spec=dict(kind="torque_driver",handle=handle,rotor=rotor,joint=clutch,root_joint=root,
        nominal_torque_nm=.4,manufacturer_tolerance_fraction=.10,
        total_mass_kg=.1089,driver_mass_kg=.0909,bit_mass_kg=.018,
        driver_length_m=.205,driver_width_m=.037,drive_across_flats_m=.004,
        bit_centre_local_m=[-.017,0,-.097],nominal_bit_bore_m=.015,
        model="native MuJoCo passive Coulomb friction clutch; no detent click dynamics",
        jaw_geometry="estimated open crescent; loaded engagement not yet qualified")
    a.hardware[name]=spec
    return spec

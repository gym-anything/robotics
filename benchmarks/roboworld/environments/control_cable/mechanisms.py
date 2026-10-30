"""Contact-built hardware. Permanent manufactured joints are not grasp welds.

All dimensions are SI. Connector local +Z points out of the socket toward the
cable. Coloured/contact surfaces are the collision surfaces, not hidden boxes.
These components are candidates until their corresponding fixtures pass.
"""
import os
import math
from pathlib import Path
import numpy as np
import warp as wp
import trimesh
from .geometry import annulus_surface,m12_external,m12_internal,newton_mesh

def perforated_insert(contact_clearance_radius=.00065):
    from shapely.geometry import Point,box
    shape=Point(0,0).buffer(.00425,quad_segs=32)
    shape=shape.difference(box(.00350,-.00060,.0045,.00060))
    for x,y in pin_positions():
        shape=shape.difference(Point(x,y).buffer(contact_clearance_radius,quad_segs=16))
    m=trimesh.creation.extrude_polygon(shape,.005,engine="earcut")
    m.apply_translation((0,0,-.002));return m

def pin_positions():
    # Four-position A-coded contact pattern; node numbers remain explicit.
    return [(r*math.cos(a),r*math.sin(a)) for r,a in
            ((.0025,math.radians(a)) for a in (225,135,45,315))]

def cable_outlet(kind,config):
    """Manufactured strain-relief position and outward axis in connector frame.

    Angled M12 envelope: Phoenix Contact 1415606 drawing, 27.4 x 34.5 mm,
    11 mm boot and 14.8 mm nut. The 90-degree boot is a reduced solid model;
    internal pin/thread geometry is shared, not an invented new mating family.
    """
    if kind=="m12" and config.sensor_connector_style=="angled":
        return np.array([-.0271,0,.0159]),np.array([-1.,0,0])
    return np.array([0.,0.,{"m12":.032,"rj45":.0327,"ferrule":.0165}[kind]]),np.array([0.,0.,1.])

class Hardware:
    """Mixin for an Assembly; every moving component is owned by native VBD."""
    def _hardware_init(self):
        self.hardware={};self.electrodes={};self.mechanism_joints={}
        self._sdf_cache=Path(os.environ.get("XDG_CACHE_HOME",Path.home()/".cache"))/"gar"/"control-cable-sdf"
        self._sdf_cache.mkdir(parents=True,exist_ok=True)

    def _free(self,name,transform):
        b=self.builder;j=b.joint_count
        body=b.add_body(xform=transform,label=name)
        self.cable_bodies.append(body);self.cable_joints.extend(range(j,b.joint_count))
        self.objects[name]=body
        return body

    def _mass(self,body,mass):
        b=self.builder
        if b.body_mass[body]<=0:raise ValueError("Physical solid needed before mass normalization")
        b.body_inertia[body]*=mass/b.body_mass[body]
        b.body_mass[body]=mass

    def _mesh(self,name,body,m,transform=None,material="nickel",sdf=False,cfg=None):
        # Convex solid hardware should have one convex contact manifold, not
        # independent contacts on the internal edges of its triangulated faces.
        # Never convexify a socket, thread, perforated insert or other cavity.
        convex=self.config.convex_hardware and not sdf and m.is_convex
        if convex and len(m.vertices)>64:raise ValueError("Do not approximate the authored convex hardware")
        add=self.builder.add_shape_convex_hull if convex else self.builder.add_shape_mesh
        i=add(body,xform=transform or wp.transform_identity(),
            mesh=newton_mesh(m,sdf=sdf,cache=str(self._sdf_cache),device=self.config.device),
            cfg=cfg or self.shape_config(ke=5e6,kd=5,mu=.15,gap=.00002,density=7000),label=name)
        self.materials[i]=material
        return i

    def _cylinder(self,name,body,radius,length,position=(0,0,0),*,material="polycarbonate",cfg=None):
        i=self.builder.add_shape_cylinder(body,xform=wp.transform(position,wp.quat_identity()),
            radius=radius,half_height=length/2,cfg=cfg or self.shape_config(),label=name)
        self.materials[i]=material
        return i

    def _world(self,frame,p,q=None):
        return wp.transform_multiply(frame,wp.transform(p,q or wp.quat_identity()))

    def m12_socket(self,name,frame):
        b=self.builder;cfg=self.shape_config(ke=5e6,kd=5,mu=.15,gap=.00002,density=7000)
        shell=self._mesh(name+"_thread",-1,m12_external(),frame,sdf=True,cfg=cfg)
        # Solid key protrudes into the otherwise open circular mating cavity.
        self.box(name+"_key",wp.transform_get_translation(self._world(frame,(.00397,0,-.0015))),
            (.00042,.00040,.0015),rotation=wp.transform_get_rotation(frame),cfg=cfg,material="polycarbonate")
        i=b.add_shape_cylinder(-1,xform=self._world(frame,(0,0,-.0045)),radius=.00435,
            half_height=.0015,cfg=self.shape_config(),label=name+"_carrier")
        self.materials[i]="polycarbonate"
        pads=[]
        for n,(x,y) in enumerate(pin_positions(),1):
            i=b.add_shape_cylinder(-1,xform=self._world(frame,(x,y,-.00125)),radius=.00045,
                half_height=.00175,cfg=self.shape_config(ke=40000,kd=.2,mu=.22,gap=.00002,density=8800),
                label=f"{name}_pin_{n}")
            self.materials[i]="copper";self.electrodes[i]=f"{name}.{n}";pads.append(i)
        self.hardware[name]=dict(kind="m12_socket",frame=np.asarray(frame).tolist(),electrodes=pads,thread_shape=shell)
        return self.hardware[name]

    def m12_plug(self,name,frame):
        b=self.builder;main=self._free(name,frame)
        anchor,exit_axis=cable_outlet("m12",self.config)
        if self.config.sensor_connector_style=="straight":
            self._cylinder(name+"_overmold",main,.0055,.027,(0,0,.0185),material="cable_black",
                cfg=self.shape_config(mu=.65,density=1200))
        else:
            self._cylinder(name+"_elbow_stem",main,.0055,.0109,(0,0,.01045),material="cable_black",
                cfg=self.shape_config(mu=.65,density=1200))
            # Actual collision geometry joins the stem to a perpendicular boot.
            q=wp.quat_from_axis_angle(wp.vec3(0,1,0),-math.pi/2)
            i=b.add_shape_cylinder(main,xform=wp.transform((-.01355,0,.0159),q),
                radius=.0055,half_height=.01355,
                cfg=self.shape_config(mu=.65,density=1200),label=name+"_angled_boot")
            self.materials[i]="cable_black"
        self._mesh(name+"_keyed_carrier",main,perforated_insert(.00085 if self.config.m12_spring_contacts else .00065),material="polycarbonate",
            sdf=True,cfg=self.shape_config(ke=150000,kd=.4,mu=.25,gap=.00002,density=1200))
        pads=[]
        for n,(x,y) in enumerate(pin_positions(),1):
            lead=self.config.m12_contact_entry_lead_m
            if self.config.m12_spring_contacts:
                pads.extend(self._m12_contact_fingers(name,main,frame,n,x,y))
                continue
            if not 0<=lead<=.001:raise ValueError("M12 contact lead must lie in [0,1] mm")
            # A female contact's turned entry guides the male pin into its
            # compliant bore. A flat undersized annulus stopped pin entry at
            # plug Z=.9024 m, exactly where its lip meets the pin end.
            z=[-.002,-.002+lead,.002] if lead else [-.002,.002]
            def bore(zz,aa):
                return .00043+(.00017*max(0.,(-.002+lead-zz)/lead) if lead else 0.)
            m=annulus_surface(z,bore,lambda z,a:.00064,sectors=32)
            i=self._mesh(f"{name}_contact_{n}",main,m,wp.transform((x,y,0),wp.quat_identity()),
                material="copper",sdf=True,cfg=self.shape_config(ke=40000,kd=.2,mu=.22,gap=.00002,density=8800))
            self.electrodes[i]=f"{name}.{n}";pads.append(i)
        seal=annulus_surface([0,.0008],lambda z,a:.00425,lambda z,a:.0054,sectors=64)
        self._mesh(name+"_seal",main,seal,material="cable_black",sdf=True,
            cfg=self.shape_config(ke=180000,kd=1.,mu=.7,gap=.00002,density=1200))
        # Moulded neck visibly and physically joins insert to the overmould.
        self._cylinder(name+"_neck",main,.00425,.002,(0,0,.004),material="polycarbonate",cfg=self.shape_config(density=1200))
        self._mass(main,.014-(16*.000005 if self.config.m12_spring_contacts else 0.))
        nut=b.add_link(xform=frame,label=name+"_nut")
        self.cable_bodies.append(nut);self.objects[name+"_nut"]=nut
        self._mesh(name+"_nut_thread",nut,m12_internal(negative_entry_chamfer=self.config.m12_negative_entry_chamfer),sdf=True)
        # Retained, freely rotating manufacturing bearing; no tightening actuator.
        j=b.add_joint_revolute(main,nut,axis=(0,0,1),target_ke=0.,target_kd=0.,
            limit_lower=-1e6,limit_upper=1e6,label=name+"_nut_bearing",
            collision_filter_parent=True)
        b.add_articulation([j],label=name+"_bearing")
        self.cable_joints.append(j);self.mechanism_joints[name+"_nut"]=j
        self._mass(nut,.004)
        self.hardware[name]=dict(kind="m12_plug",body=main,nut=nut,electrodes=pads,
            cable_anchor=anchor.tolist(),cable_exit_axis=exit_axis.tolist(),
            connector_style=self.config.sensor_connector_style,
            pin_nodes=[f"{name}.{n}" for n in range(1,5)])
        return self.hardware[name]

    def _m12_contact_fingers(self,name,main,frame,n,x,y):
        """Four spring petals per socket contact; estimated CuSn6 construction.

        k=3EI/L^3 with E=118 GPa, w=.40 mm, t=.10 mm, L=3 mm:
        437 N/m per petal. Normal compliance is an actual relative DOF.
        The 20 micrometre nominal radial interference is unchanged.
        """
        b=self.builder;pads=[]
        for sector in range(4):
            angle=sector*math.pi/2
            q=wp.quat_from_axis_angle(wp.vec3(0,0,1),angle)
            local=wp.transform((x,y,0),q)
            body=b.add_link(xform=wp.transform_multiply(frame,local),label=f"{name}_contact_{n}_petal_{sector}")
            self.cable_bodies.append(body)
            # Local +X is the outward radial direction. Actual lead-in face
            # widens at the negative-Z receiving mouth.
            profile=[(.00060,-.002),(.00043,-.0014),(.00043,.002),
                (.00064,.002),(.00064,-.002)]
            vertices=[(r,t,z) for t in (-.00020,.00020) for r,z in profile]
            shape=self._mesh(f"{name}_contact_{n}_petal_{sector}",body,
                trimesh.convex.convex_hull(vertices),material="copper",
                cfg=self.shape_config(ke=40000,kd=.2,mu=.22,gap=.00002,density=8800))
            self.electrodes[shape]=f"{name}.{n}";pads.append(shape)
            stiffness=3*118e9*(.00040*.00010**3/12)/.003**3
            j=b.add_joint_prismatic(main,body,parent_xform=local,axis=(1,0,0),
                target_pos=0,target_ke=stiffness,target_kd=.015,
                limit_lower=-.00003,limit_upper=.00015,limit_ke=40000,limit_kd=.1,
                collision_filter_parent=True,label=f"{name}_contact_{n}_petal_{sector}_spring")
            b.add_articulation([j],label=f"{name}_contact_{n}_petal_{sector}_mount")
            self.cable_joints.append(j);self._mass(body,.000005)
        return pads

    def rj45_socket(self,name,frame):
        cfg=self.shape_config(ke=150000,kd=.4,mu=.28,gap=.00003,density=1200)
        def box(n,p,h,material="polycarbonate"):
            xf=self._world(frame,p)
            return self.box(name+n,wp.transform_get_translation(xf),h,
                rotation=wp.transform_get_rotation(xf),cfg=cfg,material=material)
        # 12.0 x 8.4 mm clear mating opening. Roof includes an actual latch pocket.
        box("_left",(-.00655,0,-.010),(.00055,.0050,.010))
        box("_right",(.00655,0,-.010),(.00055,.0050,.010))
        # Contact recess below the mating floor; ribs retain the standard opening.
        box("_floor_base",(0,-.0054,-.010),(.0060,.0004,.010))
        for n in range(9):
            x=(n-4)*.00102
            box("_floor_rib"+str(n),(x,-.0046,-.010),(.00025,.0004,.010))
        for x in (-.0048,.0048):
            box("_roof_side"+str(x),(x,.0047,-.010),(.0012,.0005,.010))
        box("_latch_front_lip",(0,.0051,-.0015),(.0036,.0009,.0015))
        box("_latch_back_lip",(0,.0051,-.0145),(.0036,.0009,.0055))
        box("_roof_back",(0,.007,-.006),(.0036,.0005,.003))
        box("_back",(0,0,-.0205),(.0071,.0053,.0005))
        pads=[]
        for n in range(8):
            x=(n-3.5)*.00102
            # Explicit normal compliance of a CuSn6 leaf, not a soft material
            # on a fixed obstacle. k=3EI/L^3, E=118 GPa, w=.45, t=.22, L=10 mm.
            # Normal translation is the reduced-order bending coordinate.
            xf=self._world(frame,(x,0,0))
            body=self.builder.add_link(xform=xf,label=f"{name}_spring_contact_{n+1}")
            self.cable_bodies.append(body)
            v=np.array([[-.00022,-.00445,-.006],[.00022,-.00445,-.006],
                [-.00022,-.00361,-.009],[.00022,-.00361,-.009],
                [-.00022,-.00361,-.015],[.00022,-.00361,-.015],
                [-.00022,-.00450,-.015],[.00022,-.00450,-.015],
                [-.00022,-.00450,-.006],[.00022,-.00450,-.006]])
            i=self._mesh(f"{name}_contact_{n+1}",body,trimesh.convex.convex_hull(v),
                material="copper",cfg=self.shape_config(ke=150000,kd=.1,mu=.2,gap=.00002,density=8800))
            stiffness=3*118e9*(.00045*.00022**3/12)/.010**3
            j=self.builder.add_joint_prismatic(-1,body,parent_xform=xf,axis=(0,1,0),
                target_pos=0,target_ke=stiffness,target_kd=.03,
                limit_lower=-.00065,limit_upper=.00005,limit_ke=100000,limit_kd=.1,
                collision_filter_parent=False,label=f"{name}_contact_flexure_{n+1}")
            self.builder.add_articulation([j],label=f"{name}_contact_mount_{n+1}")
            self.cable_joints.append(j)
            self._mass(body,.000010)
            self.electrodes[i]=f"{name}.{n+1}";pads.append(i)
        self.hardware[name]=dict(kind="rj45_socket",frame=np.asarray(frame).tolist(),electrodes=pads)
        return self.hardware[name]

    def rj45_plug(self,name,frame):
        b=self.builder;body=self._free(name,frame)
        cfg=self.shape_config(ke=150000,kd=.4,mu=.28,gap=.00003,density=1200)
        self.box(name+"_nose",(0,0,.00925),(.00584,.0039,.00925),body=body,cfg=cfg,material="polycarbonate")
        self.box(name+"_boot",(0,0,.0256),(.00685,.00655,.0071),body=body,
            cfg=self.shape_config(ke=60000,kd=.3,mu=.65,density=1200),material="cable_black")
        pads=[]
        for n in range(8):
            i=self.box(f"{name}_contact_{n+1}",((n-3.5)*.00102,-.00393,.006),(.00023,.00006,.003),
                body=body,cfg=self.shape_config(ke=300,kd=.01,mu=.2,gap=.00002,density=8800),material="copper")
            self.electrodes[i]=f"{name}.{n+1}";pads.append(i)
        # Reduced-order PC cantilever, not an attach/detach switch. A hinged
        # elastic beam matches the static tip stiffness 3EI/L^3.
        length,width,thickness,E=.012,.0057,.0007,2.35e9
        rest=-.14
        # Root is embedded flush in the nose, not a square shoulder above the port roof.
        anchor=(0,.0035,.002)
        hinge=self._world(frame,anchor,wp.quat_from_axis_angle(wp.vec3(1,0,0),rest))
        latch=b.add_link(xform=hinge,label=name+"_elastic_latch")
        self.cable_bodies.append(latch);self.objects[name+"_latch"]=latch
        self.box(name+"_latch_beam",(0,0,length/2),(width/2,thickness/2,length/2),body=latch,cfg=cfg,material="polycarbonate")
        # Sloped insertion face and near-vertical retaining face on the same solid.
        w=.00285;z0=.008;z1=.010
        vertices=np.array([[-w,0,z0],[w,0,z0],[-w,0,z1],[w,0,z1],
            [-w,.0003,z0],[w,.0003,z0],[-w,.0010,z1],[w,.0010,z1]])
        self._mesh(name+"_latch_tooth",latch,trimesh.convex.convex_hull(vertices),
            material="polycarbonate",cfg=cfg)
        I=width*thickness**3/12
        j=b.add_joint_revolute(body,latch,parent_xform=wp.transform(anchor,wp.quat_identity()),
            axis=(1,0,0),target_pos=rest,target_ke=3*E*I/length,target_kd=.00003,
            limit_lower=-.25,limit_upper=.15,limit_ke=2.,limit_kd=.0001,
            collision_filter_parent=True,label=name+"_latch_spring")
        b.joint_q[b.joint_q_start[j]]=rest
        b.add_articulation([j],label=name+"_latch")
        self.cable_joints.append(j);self.mechanism_joints[name+"_latch"]=j
        self._mass(body,.0064);self._mass(latch,.000196)
        self.hardware[name]=dict(kind="rj45_plug",body=body,latch=latch,electrodes=pads,
            cable_anchor=[0,0,.0327],latch_tip_stiffness_n_m=3*E*I/length**3)
        return self.hardware[name]

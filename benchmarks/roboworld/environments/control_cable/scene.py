"""Factory workcell geometry and product-scale flexible harness inventory."""
import math
import numpy as np
import warp as wp
from .physics import Assembly
from .mechanisms import Hardware,cable_outlet
from .config import CableSpec

BENCH_DRAPES={"supply_brown":(-.25,.38),"supply_blue":(-.10,.30)}

class Workcell(Hardware,Assembly):
    def add_world(self,robots):
        self._hardware_init()
        self.labels=[];self.robot_bases={}
        z=self.config.table_height_m
        half_width=self.config.cabinet_width_m/2
        if robots:
            for name,y in (("left",-.55),("right",.55)):
                self.add_robot(name,(0,y,z))
                self.robot_bases[name]=[0,y,z]
                self.box(name+"_pedestal",(0,y,z/2),(.12,.12,z/2),material="paint_blue")
                self.box(name+"_pedestal_foot",(0,y,.018),(.19,.19,.018),material="steel")
                for dx in (-.145,.145):
                    for dy in (-.145,.145):
                        self._static_cylinder(name+"_anchor",(.0+dx,y+dy,.043),.008,.014,"nickel")
        # Cabinet service bench remains separate from the arm pedestals.
        if robots:
            from .cameras import install
            install(self)
        self.box("service_bench",(.46,0,z-.025),(.29,half_width+.03,.025),material="laminate")
        for x in (.21,.71):
            for y in (-.33,.33):
                self.box("bench_leg", (x,y,(z-.05)/2),(.025,.025,(z-.05)/2),material="aluminum")
        self.builder.add_ground_plane(cfg=self.shape_config(),label="factory_floor")
        self.materials[self.builder.shape_count-1]="concrete"
        # Room-scale envelope: factory architecture, not a distant HDRI stand-in.
        self.box("back_wall",(2.9,0,1.65),(.06,3.5,1.65),material="wall")
        self.box("side_wall",(0,3.3,1.65),(3.,.06,1.65),material="wall")
        self.box("wall_baseboard",(2.82,0,.08),(.02,3.5,.08),material="steel")
        for y in (-2.4,-1.2,0,1.2,2.4):
            self.box("wall_column",(2.78,y,1.65),(.075,.055,1.65),material="steel")
        for y in (-1.7,1.7):
            self.box("walkway_edge",(.6,y,.001),(1.9,.025,.001),material="safety_yellow")
        # Open control enclosure, DIN rail, ducts and service labels.
        self.box("cabinet_back",(.678,0,1.10),(.006,half_width,.35),material="cabinet")
        for y in (-half_width,half_width):
            self.box("cabinet_side",(.56,y,1.10),(.12,.006,.35),material="cabinet")
        for zz in (.756,1.444):
            self.box("cabinet_edge",(.56,0,zz),(.12,half_width,.006),material="cabinet")
        self.box("galvanized_mount_plate",(.655,0,1.10),(.002,half_width-.02,.32),material="aluminum")
        for zz in (.89,1.12,1.33):
            self.box("din_rail_web",(.642,0,zz),(.003,half_width-.07,.0175),material="nickel")
            for dz in (-.016,.016):
                self.box("din_rail_lip",(.636,0,zz+dz),(.006,half_width-.07,.0015),material="nickel")
        # Door opened to the side, still a solid physical obstacle.
        # Rotate about the actual hinge, including in the wider enclosure.
        q=wp.quat_from_axis_angle(wp.vec3(0,0,1),-math.pi/2)
        self.box("cabinet_open_door",(.442-half_width,half_width,1.10),(.008,half_width,.35),rotation=q,material="cabinet")
        for zz in (.86,1.33):
            self._static_cylinder("door_hinge",(.442,half_width,zz),.009,.06,"nickel")
        for y in (-(half_width-.055),half_width-.055):
            self.box("duct_back",(.617,y,1.10),(.008,.025,.27),material="duct")
            for zslot in np.arange(.845,1.371,.018):
                for side in (-1,1):
                    self.box("duct_comb",(.592,y+side*.025,zslot),(.025,.002,.005),material="duct")
        # Module casings have actual port apertures and empty insertion volumes.
        # Solid visual-box colliders would block a 20 mm RJ45 socket at just
        # 5 mm depth, so the body shell surrounds, rather than fills, it.
        shift=float(np.random.default_rng(self.config.seed).uniform(-.010,.010))+self.config.sensor_port_offset_m
        self.module_case("io_module",(.573,-.115,1.135),(.062,.12,.20),
            [(0,.015+shift,.016,.016),(0,-.045+shift,.016,.016)],"paint_blue")
        self.module_case("network_switch",(.571,.115,1.135),(.064,.112,.20),
            [(-.015,.020,.011,.0145),(.025,.020,.011,.0145)],"steel")
        self.module_case("plc",(.545,.23,1.02),(.094,.08,.15),
            [(0,.020,.011,.0145)],"paint_blue")
        self.label("PLC  /  CPU",(.542,.23,1.078),.065,.010)
        self.box("power_supply",(.602,.19,.866),(.033,.055,.055),material="aluminum")
        for yy in np.linspace(.145,.23,11):
            self.box("supply_vent",(.567,yy,.890),(.001,.002,.017),material="robot_dark")
        self.label("STATION 07 / INSPECTION",(.431,0,1.414),.39,.024)
        if self.config.task_revision>=2:
            for text,z in (("X1  SENSOR A",1.15),("X2  SENSOR B",1.09)):
                self.label(text,(.570,-.115,z+shift+.019),.105,.018)
            self.label("NETWORK SWITCH",(.568,.115,1.215),.10,.016)
            for text,y in (("LAN1",.100),("LAN2",.140)):
                self.label(text,(.568,y,1.1683),.022,.008)
        else:
            self.label("X1  SENSOR A",(.570,-.115,1.215),.105,.018)
            self.label("X2  SENSOR B",(.570,-.115,1.071),.105,.018)
            self.label("LAN1  CONTROLLER",(.568,.115,1.215),.10,.016)
        self.label("24 VDC  /  ISOLATE BEFORE SERVICE",(.568,.19,.836),.115,.011)
        # Aluminum-profile conveyor and sensor bracket, beyond the cabinet.
        for x in (.85,1.25):
            self.box("conveyor_extrusion",(x,0,.79),(.020,1.0,.04),material="aluminum")
            self.box("extrusion_slot",(x-.0205,0,.79),(.001,.985,.004),material="robot_dark")
        for y in (-.91,.91):
            for x in (.85,1.25):
                self.box("conveyor_leg",(x,y,.365),(.025,.025,.365),material="aluminum")
            self.box("conveyor_crossbeam",(1.05,y,.73),(.22,.025,.025),material="aluminum")
        self.box("conveyor_belt",(1.05,0,.833),(.18,.93,.007),material="belt")
        for y in np.linspace(-.94,.94,18):
            rotation=wp.quat_from_axis_angle(wp.vec3(0,1,0),math.pi/2)
            i=self.builder.add_shape_cylinder(-1,xform=wp.transform((1.05,float(y),.815),rotation),
                radius=.021,half_height=.197,cfg=self.shape_config(),label="conveyor_roller")
            self.materials[i]="nickel"
        self.box("sensor_bracket_foot",(.839,-.43,.949),(.040,.023,.010),material="steel")
        self.box("sensor_bracket_upright",(.873,-.43,.877),(.006,.023,.062),material="steel")
        self.module_case("photoeye",(.7795,-.43,1.003),(.065,.027,.0875),
            [(0,-.027,.016,.016)],"safety_yellow")
        self.box("photoeye_lens",(.845,-.43,1.003),(.001,.010,.023),material="red_lens")
        self.label("B1  PHOTOEYE",(.788,-.43,1.05),.08,.013)
        self.box("conveyor_motor",(1.31,.81,.71),(.08,.095,.09),material="paint_blue")
        self.box("guard_post",(1.45,0,.98),(.025,1.04,.018),material="safety_yellow")
        for y in (-.94,.94):
            self.box("guard_upright",(1.45,y,.48),(.025,.025,.48),material="safety_yellow")
        # A service trolley with realistic lipped upper tray, lower shelf and wheels.
        for zz in (.36,.745):
            self.box("cart_tray",(.25,-1.04,zz),(.31,.46,.012),material="cabinet")
        for yy in (-1.51,-.57):
            self.box("cart_lip",(.25,yy,.763),(.325,.01,.028),material="cabinet")
        for xx in (-.07,.57):
            self.box("cart_lip",(xx,-1.04,.763),(.01,.46,.028),material="cabinet")
            for yy in (-1.44,-.64):
                self.box("cart_upright",(xx,yy,.435),(.015,.015,.305),material="steel")
                self._static_cylinder("cart_caster",(xx,yy,.075),.055,.035,"cable_black",
                    rotation=wp.quat_from_axis_angle(wp.vec3(1,0,0),math.pi/2))
        # Existing protected conduit and wiring is physical, not painted on.
        for y in (-.245,.25):
            for zc in np.arange(.825,1.35,.02):
                i=self.builder.add_shape_capsule(-1,xform=wp.transform((.598,y,float(zc)),wp.quat_identity()),
                    radius=.0022,half_height=.01,cfg=self.shape_config(mu=.55),label="protected_wiring")
                self.materials[i]="cable_blue" if y>0 else "cable_brown"
        # A clipped paper schedule mounted to the trolley, not floating text.
        self.box("schedule_clipboard",(.25,-1.522,.608),(.174,.003,.126),material="laminate")
        self.box("schedule_clip",(.25,-1.528,.724),(.035,.002,.012),material="nickel")
        self.labels.append(dict(text=f"STATION 07 - WIRING SCHEDULE\n\nW1  SENSOR: {self.config.sensor_port} <-> B1\nW2  NETWORK: LAN1 <-> PLC\nW3  BROWN: SUPPLY L+ -> I/O L+\nW4  BLUE: SUPPLY M -> I/O M\n\nISOLATE / CONNECT / SECURE / TEST",
            position=[.25,-1.526,.607],width=.325,height=.225,normal=[0,-1,0]))

    def module_case(self,name,front,dimensions,ports,material):
        """Folded 2 mm housing with true port holes; coordinates are metres.

        ports=(dy,dz,width_y,height_z) on the service face normal to -X.
        The case interior remains available to complete socket geometry.
        """
        from shapely.geometry import box as profile_box
        import trimesh
        from .geometry import newton_mesh
        x,y,z=front;depth,width,height=dimensions;t=.002
        profile=profile_box(-height/2,-width/2,height/2,width/2)
        for dy,dz,wy,hz in ports:
            profile=profile.difference(profile_box(dz-hz/2,dy-wy/2,dz+hz/2,dy+wy/2))
        shape=trimesh.creation.extrude_polygon(profile,t,engine="earcut")
        shape.apply_translation((0,0,-t))
        frame=wp.transform(front,wp.quat_from_axis_angle(wp.vec3(0,1,0),-math.pi/2))
        cfg=self.shape_config(ke=150000,kd=.4,mu=.35,gap=.00003)
        self._mesh(name+"_face",-1,shape,frame,material=material,sdf=True,cfg=cfg)
        self.box(name+"_back",(x+depth-t/2,y,z),(t/2,width/2,height/2),material=material,cfg=cfg)
        for sign in (-1,1):
            self.box(name+"_side",(x+depth/2,y+sign*(width-t)/2,z),
                (depth/2,t/2,height/2),material=material,cfg=cfg)
            self.box(name+"_end",(x+depth/2,y,z+sign*(height-t)/2),
                (depth/2,(width-2*t)/2,t/2),material=material,cfg=cfg)
        # Visible and collidable face fasteners outside all port openings.
        for yy in (-width*.40,width*.40):
            for zz in (-height*.44,height*.44):
                self._static_cylinder(name+"_screw",(x-.001,y+yy,z+zz),.0014,.002,"nickel",
                    rotation=wp.quat_from_axis_angle(wp.vec3(0,1,0),math.pi/2))

    def _static_cylinder(self,name,p,radius,length,material,rotation=None):
        i=self.builder.add_shape_cylinder(-1,xform=wp.transform(p,rotation or wp.quat_identity()),
            radius=radius,half_height=length/2,cfg=self.shape_config(),label=name)
        self.materials[i]=material
        return i

    def label(self,text,position,width,height):
        self.labels.append(dict(text=text,position=list(position),width=width,height=height,
            normal=[-1,0,0]))

    def add_cable_fixture(self):
        self._hardware_init()
        self.fixed_edges=[];self.cable_inventory={}
        outward=wp.quat_from_axis_angle(wp.vec3(0,1,0),-math.pi/2)
        rng=np.random.default_rng(self.config.seed)
        shift=float(rng.uniform(-.010,.010))+self.config.sensor_port_offset_m
        sensor_outward=wp.transform_get_rotation(wp.transform_multiply(
            wp.transform((0,0,0),outward),wp.transform((0,0,0),
                wp.quat_from_axis_angle(wp.vec3(0,0,1),self.config.sensor_socket_roll_rad))))
        for name,p in (("X1",(.570,-.115,1.15+shift)),("X2",(.570,-.115,1.09+shift)),
                       ("B1",(.777,-.43,.976))):
            self.m12_socket(name,wp.transform(p,sensor_outward))
        for name,p in (("LAN1",(.568,.10,1.155)),("LAN2",(.568,.14,1.155)),
                       ("PLC",(.542,.23,1.04))):
            self.rj45_socket(name,wp.transform(p,outward))
        self.add_loose_harness("sensor",(.25,-1.27,.768),1.5,.0047,.04,
            .135,.195,"m12",100)
        self.add_loose_harness("network",(.25,-.84,.768),self.config.network_length_m,.0053,.036,
            .09,.14,"rj45",80)
        from .terminal import add_terminal,add_operating_tool
        from scipy.spatial.transform import Rotation
        # Narrow 5.2 mm dimension follows the DIN rail. The release aperture
        # and wire mouth both face the cabinet service side (-world X).
        terminal_rotation=Rotation.from_matrix([[0,0,-1],[-1,0,0],[0,1,0]]).as_quat()
        for name,y in (("SUPPLY_L",.06),("SUPPLY_M",.054),
                       ("IO_L",-.08),("IO_M",-.086)):
            add_terminal(self,name,wp.transform((.605,y,.918),wp.quat(*terminal_rotation)))
            self.label("L+" if name.endswith("_L") else "M",(.602,y,.942),.0052,.009)
        self.label("SUPPLY",(.602,.057,.955),.025,.006)
        self.label("I/O",(.602,-.083,.955),.025,.006)
        # Prepared replacement leads stored on the trolley's lower shelf
        # (task revision 2: laid on the service bench, see bench_drape).
        # LAPP H05V-K 1 mm2: 2.4..2.8 mm OD, 15 kg/km. EI/GJ are documented
        # stranded-wire estimates, not the stiffness of a solid copper rod.
        for name,y,color in (("supply_brown",-1.17,"cable_brown"),("supply_blue",-.89,"cable_blue")):
            self.add_loose_harness(name,(.25,y,.378),.6,.0026,.015,
                .055,.085,"ferrule",40,color=color,bending=.00030,torsional=.00015)
        add_operating_tool(self,"terminal_tool",wp.transform((.51,-1.07,.768),
            wp.quat_from_axis_angle(wp.vec3(1,0,0),math.pi/2)))
        from .routing import install
        from .torque_tool import add_driver
        # Loose product-scale tool on an unused part of the cart. Native
        # .4 Nm clutch; no tool-to-connector attachment or automatic actuation.
        add_driver(self,"torque_driver",wp.transform((.03,-.97,.777),
            wp.quat_from_axis_angle(wp.vec3(1,0,0),np.pi/2)))
        self.label("TORQUE DRIVER / 0.4 N m",(.04,-.575,.798),.14,.009)
        install(self)
        from .protected import install as install_protected_uplink
        install_protected_uplink(self)
        from .commissioning import install as install_commissioning
        install_commissioning(self)
        self.labels.append(dict(text="ROUTES: W1/G1  W2/G2  W3/G3  W4/G4",
            position=[.25,-1.529,.494],width=.32,height=.016,normal=[0,-1,0]))

    def bench_drape(self,y,x_turn,*,diameter,bottom=.60,radius=.025,turn=.04):
        """Revision-2 lead layout: end 0 hangs over the service bench's front edge
        (x=0.17), the rest lies on the bench along +x to x_turn, then along +y."""
        edge=.17;top=self.config.table_height_m+diameter/2+.00005
        a=np.linspace(math.pi,math.pi/2,40);b=np.linspace(-math.pi/2,0,40)
        return np.concatenate((np.linspace([edge-radius,y,bottom],[edge-radius,y,top-radius],60),
            np.column_stack((edge+radius*np.cos(a),np.full(len(a),y),top-radius+radius*np.sin(a))),
            np.linspace([edge,y,top],[x_turn-turn,y,top],60),
            np.column_stack((x_turn-turn+turn*np.cos(b),y+turn+turn*np.sin(b),np.full(len(b),top))),
            np.linspace([x_turn,y+turn,top],[x_turn,y+.6,top],120)))

    def add_loose_harness(self,name,centre,length,diameter,line_mass,r0,r1,kind,segments,
                          *,color=None,bending=.00070,torsional=.00030):
        if self.config.task_revision>=2 and name in BENCH_DRAPES:
            path=self.bench_drape(*BENCH_DRAPES[name],diameter=diameter)
            arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
            points=np.column_stack([np.interp(np.linspace(0,length,segments+1),arc,path[:,i]) for i in range(3)])
        else:
            # Planar manufacturer-scale coil; radius remains above its installation
            # limit. Uniform arclength resampling keeps cable mass/stiffness local.
            turns=length/(math.pi*(r0+r1))
            theta=np.linspace(0,2*math.pi*turns,2001)
            radii=np.linspace(r0,r1,len(theta))
            points=np.column_stack((radii*np.cos(theta),radii*np.sin(theta),np.zeros(len(theta))))
            lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            # Match requested cable length, not a silent number-of-turns approximation.
            points*=length/lengths[-1]
            lengths*=length/lengths[-1]
            points=np.column_stack([np.interp(np.linspace(0,length,segments+1),lengths,points[:,i]) for i in range(3)])
            if name=="network":
                angle=self.config.network_coil_rotation_rad
                cs,sn=math.cos(angle),math.sin(angle)
                points=points@np.array([[cs,sn,0],[-sn,cs,0],[0,0,1]])
            points+=np.asarray(centre)
        h=length/segments
        first_shape=self.builder.shape_count
        bodies,joints=self.builder.add_rod(points.tolist(),radius=diameter/2,
            body_frame_origin="com",label=name+"_cable",
            cfg=self.shape_config(mu=.45,density=1200),stretch_stiffness=100000/h,
            shear_stiffness=100000/h,bend_stiffness=bending/h,twist_stiffness=torsional/h,
            stretch_damping=.05,shear_damping=.05,bend_damping=2e-5/h,twist_damping=1e-5/h)
        self.cable_bodies.extend(bodies);self.cable_joints.extend(joints)
        self.objects[name+"_cable"]=bodies
        for body in bodies:self._mass(body,line_mass*h)
        for i in range(first_shape,self.builder.shape_count):
            self.materials[i]=color or ("cable_black" if name=="sensor" else "cable_green")
        from scipy.spatial.transform import Rotation
        for end,point,tangent,rod in ((0,points[0],points[1]-points[0],bodies[0]),
                                     (1,points[-1],points[-2]-points[-1],bodies[-1])):
            tangent=tangent/np.linalg.norm(tangent)
            anchor,exit_axis=cable_outlet(kind,self.config)
            if kind=="m12" and self.config.sensor_connector_style=="angled":
                # Keep the whole L-shaped connector in the trolley plane. Its
                # outlet -X, not its mating +Z, follows the cable tangent.
                xx=-tangent;yy=np.array([0.,0.,1.]);zz=np.cross(xx,yy)
                rotation=Rotation.from_matrix(np.column_stack((xx,yy,zz)))
            else:rotation=Rotation.align_vectors([tangent],[[0,0,1]])[0]
            q=rotation.as_quat()
            origin=point-rotation.apply(anchor)
            from .terminal import add_ferrule
            make={"m12":self.m12_plug,"rj45":self.rj45_plug,
                  "ferrule":lambda label,frame:add_ferrule(self,label,frame)}[kind]
            connector=make(f"{name}_end_{end}",wp.transform(origin,wp.quat(*q)))
            # Bonded strain relief is a permanent manufactured attachment.
            rod_q=np.asarray(self.builder.body_q[rod])
            world=wp.transform(point,wp.quat(*q))
            parent=wp.transform(anchor,wp.quat_identity())
            child=wp.transform_multiply(wp.transform_inverse(wp.transform(rod_q[:3],rod_q[3:])),world)
            j=self.builder.add_joint_fixed(connector["body"],rod,parent_xform=parent,
                child_xform=child,label=f"{name}_strain_relief_{end}",collision_filter_parent=True)
            self.builder.add_articulation([j],label=f"{name}_bond_{end}")
            self.cable_joints.append(j)
        from .electrical import Edge
        pins={"m12":4,"rj45":8,"ferrule":1}[kind]
        # 17.24 nOhm m copper resistivity; 1 mm2 leads, 0.34 mm2 sensor,
        # 0.205 mm2 (24 AWG) network are declared nominal conductor areas.
        area={"m12":.34e-6,"rj45":.205e-6,"ferrule":1e-6}[kind]
        for pin in range(1,pins+1):
            suffix="conductor" if kind=="ferrule" else str(pin)
            self.fixed_edges.append(Edge(f"{name}_end_0.{suffix}",f"{name}_end_1.{suffix}",1.724e-8*length/area))
        self.cable_inventory[name]=dict(length_m=length,diameter_m=diameter,line_mass_kg_m=line_mass,
            bending_rigidity_nm2=bending,torsional_rigidity_nm2=torsional,segments=segments,kind=kind)

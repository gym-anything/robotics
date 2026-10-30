"""Physical TEST switch and contact-dependent low-voltage functional check.

The optical emitter-disable test follows SICK WL24-2 test-input semantics.
Electrical logic is a lumped digital model, not Ethernet waveforms or arcing.
No commissioning API action exists: depress the actual spring-return switch.
"""
import numpy as np
import warp as wp
from .geometry import annulus_surface
from .electrical import Edge

def install(a):
    b=a.builder;rotation=wp.quat_from_axis_angle(wp.vec3(0,1,0),-np.pi/2)
    frame=wp.transform((.570,-.225,1.280),rotation)
    cfg=a.shape_config(ke=60000,kd=1.,mu=.45,gap=.00002,density=1200)
    bezel=annulus_surface([-.009,0],lambda z,t:.011,lambda z,t:.014,sectors=64)
    a._mesh("test_switch_bezel",-1,bezel,frame,sdf=True,cfg=cfg,material="robot_dark")
    a.box("test_switch_mount",(.650,-.225,1.280),(.003,.032,.035),material="cabinet")
    housing=annulus_surface([-.079,-.009],lambda z,t:.0044,lambda z,t:.014,sectors=64)
    a._mesh("test_switch_housing",-1,housing,frame,sdf=True,cfg=cfg,
        material="robot_dark")
    body=b.add_link(xform=frame,label="commissioning_test_button")
    a.cable_bodies.append(body);a.objects["commissioning_test_button"]=body
    a._cylinder("test_button_face",body,.0105,.003,(0,0,.0015),material="cable_green",cfg=cfg)
    a._cylinder("test_button_shaft",body,.004,.018,(0,0,-.0075),material="polycarbonate",cfg=cfg)
    # Schneider XB5AA31: state change 2.6 mm at 3.8 N, total stroke 4.3 mm.
    # Linear reduced spring interpolates that product operating point; 12 g
    # moving mass and damping are explicitly estimated.
    joint=b.add_joint_prismatic(-1,body,parent_xform=frame,axis=(0,0,1),
        target_pos=0,target_ke=3.8/.0026,target_kd=.25,
        limit_lower=-.0043,limit_upper=0,limit_ke=100000,limit_kd=1.,
        collision_filter_parent=False,label="test_switch_spring")
    b.add_articulation([joint],label="test_switch_mount")
    a.cable_joints.append(joint);a._mass(body,.012)
    a.commissioning_switch=dict(body=body,joint=joint,q=b.joint_q_start[joint],
        change_travel_m=.0026,total_travel_m=.0043,operating_force_n=3.8)
    a.label("COMMISSION / TEST",(.568,-.225,1.325),.100,.011)
    a.label("FIELD SUPPLY ISOLATED UNTIL TEST",(.568,-.225,1.244),.113,.007)
    for signal,y,title in (("power",-.250,"PWR"),("pass_",-.225,"PASS"),("fail",-.200,"FAIL")):
        xf=wp.transform((.569,y,1.230),rotation)
        shape=b.add_shape_cylinder(-1,xform=xf,radius=.003,half_height=.001,
            cfg=cfg,label="indicator_"+signal)
        a.materials[shape]="indicator_"+signal
        a.label(title,(.567,y,1.216),.020,.006)
    led=a.box("photoeye_status_led",(.8455,-.419,1.027),(.001,.002,.002),material="indicator_sensor")
    # Manufactured I/O electronics: field power and return reach the sensor
    # through its real four-pin interface, not through task-name shortcuts.
    a.fixed_edges.extend((Edge("IO_L.conductor",a.config.sensor_port+".1",.02),
                          Edge("IO_M.conductor",a.config.sensor_port+".3",.02)))
    # Opposing retroreflector on a physically mounted bracket.
    a.box("reflector_bracket",(1.280,-.430,.940),(.006,.040,.105),material="steel")
    reflector=a.box("photoeye_reflector",(1.272,-.430,1.003),(.002,.025,.025),material="nickel")
    a.photoeye=dict(origin_m=[.8462,-.430,1.003],direction=[1.,0.,0.],
        reflector_shape=reflector,range_m=.60)

class OpticalPath:
    def __init__(self,env):
        self.env=env;m=env.model;d=m.device
        self.lock=wp.zeros(1,dtype=wp.int32,device=d)
        self.distance=wp.zeros(1,dtype=float,device=d)
        self.shape=wp.zeros(1,dtype=int,device=d);self.body=wp.zeros(1,dtype=int,device=d)
        self.empty_int=wp.empty(0,dtype=int,device=d)
        self.empty_vec=wp.empty(0,dtype=wp.vec3,device=d)
    def read(self):
        from newton._src.geometry.raycast import raycast_kernel
        e=self.env;m=e.model;s=e.assembly.photoeye
        self.lock.zero_();self.distance.fill_(s["range_m"])
        self.shape.fill_(-1);self.body.fill_(-1)
        wp.launch(raycast_kernel,dim=m.shape_count,inputs=[e.state.body_q,m.shape_body,
            m.shape_transform,m.shape_type,m.shape_scale,m.shape_source_ptr,
            wp.vec3(*s["origin_m"]),wp.vec3(*s["direction"]),self.lock,
            self.distance,self.shape,self.body,self.empty_int,self.empty_vec,self.empty_int],
            device=m.device)
        shape=int(self.shape.numpy()[0])
        return dict(clear=shape==s["reflector_shape"],hit_shape=shape,
                    distance_m=float(self.distance.numpy()[0]))

class Commissioning:
    def __init__(self,env):
        self.optics=OpticalPath(env);self.last_pressed=False
        self.press_since=None;self.started=None;self.samples={}
        self.status="isolated";self.failure=None;self.last_output=None
        self.transitions=[];self.signals=dict(power=False,sensor=False,pass_=False,fail=False)
    def update(self,env,wiring_correct,short,*,optical=None,sustained_loss=None):
        """sustained_loss (revision 2): debounced circuit loss that ends a PASS.

        None keeps the revision-1 rule: any single raw tick ends it.
        """
        t=env.tick*env.config.dt
        stroke=-float(env.read_array(env.state.joint_q)[env.assembly.commissioning_switch["q"]])
        down=stroke>=.0026
        if down:
            if self.press_since is None:self.press_since=t
        else:self.press_since=None
        pressed=down and t-self.press_since>=.020
        if pressed and not self.last_pressed:
            self.started=t;self.samples={};self.failure=None;self.status="testing"
        self.last_pressed=pressed
        power=self.status in ("testing","passed") and not short
        rplus=env.network.path_resistance("SUPPLY_L.conductor","IO_L.conductor")
        rminus=env.network.path_resistance("SUPPLY_M.conductor","IO_M.conductor")
        voltage=0. if not power or rplus is None or rminus is None else 24*120/(120+rplus+rminus)
        elapsed=0. if self.started is None else t-self.started
        test_input_low=self.status=="testing" and .20<=elapsed<.40
        optical=self.optics.read() if optical is None else optical
        sensor_power=power and voltage>=20.4 and env.network.connected("IO_L.conductor","B1.1") and env.network.connected("IO_M.conductor","B1.3")
        test_reaches_sensor=test_input_low and env.network.connected(env.config.sensor_port+".2","B1.2")
        sender_on=sensor_power and not test_reaches_sensor
        receiver_high=sender_on and optical["clear"]
        plc_input=receiver_high and env.network.connected("B1.4",env.config.sensor_port+".4")
        network_ok=env.network.pin_map("LAN1","PLC",8)["correct"]
        received=plc_input if network_ok else None
        if received!=self.last_output:
            self.transitions.append(dict(tick=env.tick,plc_input=received))
            self.transitions=self.transitions[-64:];self.last_output=received
        if self.status=="testing":
            if short or not wiring_correct or voltage<20.4:
                self.status="failed";self.failure="Circuit open, miswired, shorted or undervoltage"
            else:
                for name,deadline in (("emitter_on",.15),("emitter_disabled",.35),("emitter_restored",.55)):
                    if elapsed>=deadline and name not in self.samples:self.samples[name]=received
                if elapsed>=.60:
                    good=self.samples==dict(emitter_on=True,emitter_disabled=False,emitter_restored=True)
                    self.status="passed" if good else "failed"
                    self.failure=None if good else "Optical test-input/return sequence failed"
        elif self.status=="passed" and ((not wiring_correct or short) if sustained_loss is None else sustained_loss):
            self.status="failed";self.failure="Circuit lost after commissioning"
        self.signals=dict(power=self.status in ("testing","passed"),sensor=bool(receiver_high),
                          pass_=self.status=="passed",fail=self.status=="failed")
        return dict(status=self.status,passed=self.status=="passed",failure=self.failure,
            button_stroke_m=stroke,field_voltage_v=voltage,field_load_ohm=120.,
            test_input_active_low=test_input_low,sender_on=bool(sender_on),
            optical=optical,plc_received=received,samples=dict(self.samples),
            transitions=list(self.transitions),signals=dict(self.signals))

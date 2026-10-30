"""Physical wiring, routing, protected-circuit and commissioning gates.

Revision 1 (Progress) is the original evaluator of every preset except
offset-lower-service. Revision 2 (ServiceProgress, config.task_revision=2,
offset-lower-service) keeps the same physical
measurements and changes what they are taken to mean:

* Landed destination: the SCHEDULED lead's end has a direct metal-contact edge
  (>= 1 mN, electrical.py) on every pin of that destination and on no other pin
  of it. Brown lands only on SUPPLY L+/I/O L+, blue only on SUPPLY M/I/O M.
  RJ45 ends must also be latched: plug inside the 12.0 x 8.4 mm opening, latch
  tooth retaining face no more than 0.2 mm outside the front lip's inner face
  (socket z < -2.8 mm; a face there with the tooth top above the lip is in
  contact with the lip) and the tooth top at least 0.2 mm above the lip
  underside (socket y > 4.4 mm; a tooth held down under the lip rests at
  4.21 mm in soft contact, a risen tooth at 4.78 mm or more). At full seat the beam end rests on the lip and the tooth face
  reads about -5.1 mm; the tooth is a second catch about 2 mm further out.
  Ferrules must also be clamped: tip inside the 3.4 mm wire channel, at least
  8.9 mm deep (where the blade's rest face meets a ferrule on the busbar;
  seated -12.82 mm), the blade deflected >= 0.02 rad (seated 0.058) AND the
  blade in active contact with that ferrule. The operating lever deflects the
  blade the same way, so a clamp held open is not clamped.
  M12 ends need no extra predicate: an unrotated coupling nut stops about 6 mm
  short and the pins conduct only with >= 4 of 6 thread turns engaged.
* Every new circuit, landed state and L+/M isolation uses the protected
  uplink's 20 ms debounce: a raw opening must persist 20 ms to count as lost.
  A PASS ends only on such a sustained loss. The 0.6 s test itself still reads
  raw contacts. Raw per-tick values stay in the evaluation.
* Routing: the cable centreline crosses the eye inside its 11 mm hole radius and
  its signed crossings from end 0 to end 1 do not cancel (no in-and-back bight).
* Released: no robot shape, and no free tool (terminal tool, torque driver) that
  a robot shape is touching, in active contact with a cable rod, connector
  body (plugs, ferrules, nuts, petals, latches, uplink) or a supply terminal's
  clamp spring (blade, lever, pad). The TEST button and socket leaves are not.
* Hands-off dwell: all destinations landed, circuits correct, routes, LAN2
  continuity and PASS, robot released, and the eight new connector bodies below
  a 0.02 m/s 50 ms motion envelope, continuously for N ticks with N*dt >= 2 s.
  Slack rods are not required to be motionless; their envelope is reported.
* success = all gates; score = 1.0 on success, else 0.9 * (units + partial)/13
  with units = landed destinations (supply destinations 0 while L+/M is
  shorted) + routed guides + standing PASS. partial = 0.5 x insertion depth
  (insertion_fraction) of the scheduled cable's end in its own destination,
  for destinations not yet landed; never in another destination, never for
  supply destinations while shorted. After a latched LAN2 violation the score
  is x0.5 if the uplink has conducted again for at least 20 ms, x0.25 if not. Everything is read from the current state, so order, method and tools
  do not matter.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .protected import Integrity
from .commissioning import Commissioning
from .routing import inspect as inspect_routing
from .instructions import instruction

DESCRIPTION=("Restore the inspection station's control connections using the wiring schedule. "
    "Connect the sensor harness, network link, and supply leads; route W1 through G1, W2 "
    "through G2, W3 through G3 and W4 through G4. Press the physical COMMISSION/TEST "
    "button. Leave all cables released and stable for two seconds. Preserve the existing LAN2 uplink.")

class Progress:
    @staticmethod
    def description(config):
        return instruction(config) or DESCRIPTION+f" Wiring schedule: sensor {config.sensor_port} to B1; network LAN1 to PLC; brown SUPPLY_L to IO_L; blue SUPPLY_M to IO_M."

    def __init__(self,env):
        self.unassisted_since=None;self.integrity=Integrity()
        self.commissioning=Commissioning(env);self.routing={}
        self.geometry_tick=None;self.optical=None
        from .stability import MotionWindow
        self.moving=[]
        for cable in env.assembly.cable_inventory:
            self.moving.extend(env.assembly.objects[cable+"_cable"])
        self.robot_ids={i for i,n in enumerate(env.builder.body_label) if n.startswith(("left/","right/"))}
        self.cable_ids=set(env.assembly.cable_bodies)
        self.contact_shape_groups=tuple(np.array([body in group for body in env.builder.shape_body])
            for group in (self.robot_ids,self.cable_ids))
        self.motion=MotionWindow(env.config.dt,env.read_array(env.state.body_q)[self.moving,:3])

    def initial(self):
        return dict(success=False,score=0.,qualification="assisted_digital_evidence_scoped_in_dossier",
            circuit_correct=False,hands_off_seconds=0.,routing={},protected={},
            commissioning={"status":"isolated","passed":False})

    def update(self,env):
        network=env.network
        sensor=network.pin_map(env.config.sensor_port,"B1",4)
        ethernet=network.pin_map("LAN1","PLC",8)
        supply=network.connected("SUPPLY_L.conductor","IO_L.conductor")
        return_path=network.connected("SUPPLY_M.conductor","IO_M.conductor")
        short=network.connected("SUPPLY_L.conductor","SUPPLY_M.conductor") or network.connected("IO_L.conductor","IO_M.conductor")
        correct=sensor["correct"] and ethernet["correct"] and supply and return_path and not short
        protected=self.integrity.update(env)
        # Geometry-derived readings are sampled every 20 physics ticks, explicitly
        # timestamped. Raw physics/contact/electrical history still records each tick.
        if self.geometry_tick is None or env.tick-self.geometry_tick>=20:
            self.routing=inspect_routing(env.assembly,env.read_array(env.state.body_q))
            self.optical=self.commissioning.optics.read()
            self.geometry_tick=env.tick
        commissioning=self.commissioning.update(env,correct,short,optical=self.optical)
        routed=all(x["passed_through"] for x in self.routing.values()) and len(self.routing)==4
        touched=bool(env.active_contact_pairs(shape_groups=self.contact_shape_groups))
        speed=float(np.linalg.norm(env.read_array(env.state.body_qd)[self.moving,:3],axis=1).max(initial=0.))
        motion_rate=self.motion.update(env.read_array(env.state.body_q)[self.moving,:3])
        # The dwell timer uses the existing 20 ms continuity monitor. A raw
        # sub-threshold contact dropout remains visible and makes the current
        # protected gate false, but must not bypass its documented debounce by
        # restarting the entire two-second mechanical-settling interval.
        ready=correct and routed and protected["continuity_preserved"] and commissioning["passed"]
        if ready and not touched and motion_rate is not None and motion_rate<.02:
            if self.unassisted_since is None:self.unassisted_since=env.tick
        else:self.unassisted_since=None
        seconds=0. if self.unassisted_since is None else (env.tick-self.unassisted_since)*env.config.dt
        gates=dict(wiring=correct,routing=routed,protected=protected["preserved"],
            commissioning=commissioning["passed"],hands_off_stable=seconds>=2.)
        return dict(success=all(gates.values()),score=sum(gates.values())/len(gates),
            qualification="assisted_digital_evidence_scoped_in_dossier",gates=gates,sensor=sensor,network=ethernet,
            supply=supply,return_path=return_path,supply_short=short,circuit_correct=correct,
            robot_contact=touched,max_cable_speed_m_s=speed,hands_off_seconds=seconds,
            settled_motion_rate_m_s=motion_rate,motion_window_s=self.motion.seconds,
            settled_motion_definition="Maximum per-body position-envelope diagonal over every physics step in the trailing window, divided by window seconds",
            settled_motion_limit_m_s=.02,
            routing=self.routing,protected=protected,commissioning=commissioning,
            geometry_evaluation_tick=self.geometry_tick,geometry_sample_interval_ticks=20)


DEBOUNCE_S=.020
MOTION_LIMIT_M_S=.02
DWELL_S=2.
RJ45_OPENING_HALF_M=(.0060,.0042)
RJ45_DEPTH_M=.021
TOOTH_FACE_LOCAL_M=(0.,.0005,.010)
TOOTH_TOP_LOCAL_M=(0.,.0010,.010)
FRONT_LIP_INNER_Z_M=-.003
FRONT_LIP_CONTACT_TOLERANCE_M=.0002
FRONT_LIP_UNDERSIDE_Y_M=.0042
TOOTH_RISEN_MARGIN_M=.0002
WIRE_CHANNEL_HALF_M=.0017
CLAMP_LINE_Z_M=-.0089
BLADE_MIN_DEFLECTION_RAD=.02
RJ45_SEAT_DEPTH_M=.01698
FERRULE_SEAT_DEPTH_M=.01282
M12_THREAD_LENGTH_M=.006
M12_COAXIAL_M=.002
PARTIAL_ENGAGEMENT_UNIT=.5
UNIT_FACTOR=.9
VIOLATION_FACTOR=.5
UNRESTORED_VIOLATION_FACTOR=.25
SUPPLY_CABLES=("supply_brown","supply_blue")


class Sustained:
    """True only once raw True has lasted the debounce; False on any raw False."""
    def __init__(self,debounce=DEBOUNCE_S):
        self.debounce=debounce;self.since=None
    def update(self,raw,now):
        if not raw:self.since=None
        elif self.since is None:self.since=now
        return self.since is not None and now-self.since>=self.debounce


class Held:
    """True from a raw True; False only after raw False has lasted the debounce."""
    def __init__(self,debounce=DEBOUNCE_S):
        self.debounce=debounce;self.state=False;self.since=None
    def update(self,raw,now):
        if raw:self.state,self.since=True,None
        else:
            if self.since is None:self.since=now
            if now-self.since>=self.debounce:self.state=False
        return self.state


def _frame(values):
    values=np.asarray(values,float)
    return values[:3],Rotation.from_quat(values[3:])


def rj45_latched(q,plug,latch,socket):
    """Pose-only latch engagement of an RJ45 plug in a socket frame (local +Z outward)."""
    origin,rotation=socket;inverse=rotation.inv()
    p=inverse.apply(q[plug,:3]-origin)
    latch_rotation=Rotation.from_quat(q[latch,3:])
    face=inverse.apply(q[latch,:3]+latch_rotation.apply(TOOTH_FACE_LOCAL_M)-origin)
    top=inverse.apply(q[latch,:3]+latch_rotation.apply(TOOTH_TOP_LOCAL_M)-origin)
    inside=abs(p[0])<RJ45_OPENING_HALF_M[0] and abs(p[1])<RJ45_OPENING_HALF_M[1] and -RJ45_DEPTH_M<p[2]<0
    latched=bool(inside and face[2]<FRONT_LIP_INNER_Z_M+FRONT_LIP_CONTACT_TOLERANCE_M and top[1]>FRONT_LIP_UNDERSIDE_Y_M+TOOTH_RISEN_MARGIN_M)
    return latched,dict(plug_socket_z_m=float(p[2]),tooth_face_socket_z_m=float(face[2]),
                        tooth_top_socket_y_m=float(top[1]))


def ferrule_clamped(q,ferrule,terminal,spring,rest,bearing):
    """Clamp engagement of a ferrule in a spring terminal (local +Z outward).

    bearing: the terminal's blade shape is in active contact with this ferrule.
    The operating lever turns the blade the same way a ferrule does, so blade
    deflection alone cannot tell a clamp on the ferrule from a clamp held open.
    """
    origin,rotation=terminal;inverse=rotation.inv()
    tip=inverse.apply(q[ferrule,:3]-origin)
    axis=(inverse*Rotation.from_quat(q[ferrule,3:])).apply([0.,0.,1.])
    m=(inverse*Rotation.from_quat(q[spring,3:])).as_matrix()
    deflection=float((np.arctan2(m[2,1],m[1,1])-rest+np.pi)%(2*np.pi)-np.pi)
    channel=abs(tip[0])<=WIRE_CHANNEL_HALF_M and abs(tip[1])<=WIRE_CHANNEL_HALF_M and axis[2]>.9
    clamped=bool(channel and tip[2]<=CLAMP_LINE_Z_M and deflection>=BLADE_MIN_DEFLECTION_RAD and bearing)
    return clamped,dict(tip_terminal_z_m=float(tip[2]),blade_deflection_rad=deflection,blade_bears_on_ferrule=bool(bearing))


def insertion_fraction(q,body,kind,frame):
    """Depth of a connector along its own destination's insertion axis, 0..1.

    RJ45: plug inside the opening, coaxial, latch side up; nose depth over the
    16.98 mm seat. Ferrule: tip in the wire channel, coaxial; depth over the
    12.82 mm seat. M12: plug coaxial within 2 mm; thread engaged out of 6 mm
    (an unrotated nut stops on the first crest, about 6 mm out). 0 otherwise.
    """
    origin,rotation=frame;inverse=rotation.inv()
    p=inverse.apply(q[body,:3]-origin);local=inverse*Rotation.from_quat(q[body,3:])
    if local.apply([0.,0.,1.])[2]<=.9:return 0.
    if kind=="rj45":
        ok=abs(p[0])<RJ45_OPENING_HALF_M[0] and abs(p[1])<RJ45_OPENING_HALF_M[1] and -RJ45_DEPTH_M<p[2]<0 and local.apply([0.,1.,0.])[1]>0
        return float(min(1.,-p[2]/RJ45_SEAT_DEPTH_M)) if ok else 0.
    if kind=="ferrule":
        ok=abs(p[0])<=WIRE_CHANNEL_HALF_M and abs(p[1])<=WIRE_CHANNEL_HALF_M and p[2]<0
        return float(min(1.,-p[2]/FERRULE_SEAT_DEPTH_M)) if ok else 0.
    if np.hypot(p[0],p[1])>M12_COAXIAL_M:return 0.
    return float(np.clip((M12_THREAD_LENGTH_M-max(p[2],0.))/M12_THREAD_LENGTH_M,0.,1.))


class ServiceProgress(Progress):
    """Revision-2 evaluator and progress score (see module docstring)."""
    def __init__(self,env):
        from .stability import MotionWindow
        a=env.assembly;b=env.builder;labels=b.body_label
        self.integrity=Integrity();self.commissioning=Commissioning(env)
        self.routing={};self.geometry_tick=None;self.optical=None;self.since=None
        self.destinations=[]
        for cable,ports in (("sensor",(env.config.sensor_port,"B1")),("network",("LAN1","PLC")),
                            ("supply_brown",("SUPPLY_L","IO_L")),("supply_blue",("SUPPLY_M","IO_M"))):
            kind=a.cable_inventory[cable]["kind"]
            for port in ports:self.destinations.append((port,cable,kind))
        self.held={port:Held() for port,_,_ in self.destinations}
        self.circuit_held=Held();self.isolated_held=Held();self.uplink_held=Sustained()
        self.frames={port:_frame(a.hardware[port]["frame"]) for port,_,_ in self.destinations}
        self.unit_count=len(self.destinations)+len(a.routing_guides)+1
        self.connector_ids=[a.hardware[f"{cable}_end_{end}"]["body"] for cable in
                            ("sensor","network",*SUPPLY_CABLES) for end in (0,1)]
        self.moving=[i for cable in a.cable_inventory for i in a.objects[cable+"_cable"]]
        prefixes=tuple(cable+"_" for cable in a.cable_inventory)
        cable_side={i for i in a.cable_bodies if labels[i] in a.cable_inventory or labels[i].startswith(prefixes)}
        springs={port:a.hardware[port]["spring"] for port,_,kind in self.destinations if kind=="ferrule"}
        cable_side|=set(springs.values())
        shape_label=list(getattr(b,"shape_label",()))
        self.blade_port={s:port for port,body in springs.items() for s,owner in enumerate(b.shape_body)
                         if owner==body and s<len(shape_label) and shape_label[s]==port+"_blade"}
        ferrules={a.hardware[f"{cable}_end_{end}"]["body"] for cable in SUPPLY_CABLES for end in (0,1)}
        robot={i for i,n in enumerate(labels) if n.startswith(("left/","right/"))}
        self.tools=[{a.objects[n] for n in names if n in a.objects} for names in
                    (("terminal_tool",),("torque_driver_handle","torque_driver_shaft_and_bit"))]
        self.tools=[tool for tool in self.tools if tool]
        self.shape_body=np.asarray(b.shape_body)
        mask=lambda bodies:np.isin(self.shape_body,sorted(bodies))
        self.robot_mask,self.cable_mask=mask(robot),mask(cable_side)
        self.tool_mask=mask(set().union(*self.tools))
        self.blade_mask=np.isin(np.arange(len(self.shape_body)),sorted(self.blade_port))
        self.ferrule_mask=mask(ferrules)
        q=env.read_array(env.state.body_q)
        self.connector_motion=MotionWindow(env.config.dt,q[self.connector_ids,:3])
        self.motion=MotionWindow(env.config.dt,q[self.moving,:3])

    def initial(self):
        return dict(success=False,score=0.,task_revision=2,
            qualification="assisted_digital_evidence_scoped_in_dossier",circuit_correct=False,
            hands_off_seconds=0.,routing={},protected={},connections={},progress_units={},
            commissioning={"status":"isolated","passed":False})

    def _landed(self,env,q,pairs,port,cable,kind,bearing):
        a=env.assembly
        for end in (0,1):
            name=f"{cable}_end_{end}"
            if kind=="ferrule":
                if tuple(sorted((port+".conductor",name+".conductor"))) not in pairs:continue
                spring=a.hardware[port];body=a.hardware[name]["body"]
                engaged,detail=ferrule_clamped(q,body,self.frames[port],spring["spring"],
                                               spring["spring_rest_angle_rad"],(port,body) in bearing)
            else:
                pins=4 if kind=="m12" else 8
                touch=lambda i,j:tuple(sorted((f"{port}.{i}",f"{name}.{j}"))) in pairs
                if not all(touch(i,i) for i in range(1,pins+1)):continue
                if any(touch(i,j) for i in range(1,pins+1) for j in range(1,pins+1) if i!=j):continue
                if kind=="rj45":
                    engaged,detail=rj45_latched(q,a.hardware[name]["body"],a.hardware[name]["latch"],self.frames[port])
                else:engaged,detail=True,{}
            return engaged,dict(end=name,pins_in_contact=True,engaged=engaged,**detail)
        return False,dict(end=None,pins_in_contact=False,engaged=False)

    def _robot_contact(self,env):
        held=set()
        if self.tool_mask.any():
            touched={int(self.shape_body[s]) for pair in env.active_contact_pairs(
                shape_groups=(self.robot_mask,self.tool_mask)) for s in pair}
            for tool in self.tools:
                if tool&touched:held|=tool
        robot=self.robot_mask|np.isin(self.shape_body,sorted(held)) if held else self.robot_mask
        return bool(env.active_contact_pairs(shape_groups=(robot,self.cable_mask))),sorted(held)

    def update(self,env):
        network=env.network;config=env.config;now=env.tick*config.dt
        q=env.read_array(env.state.body_q)
        sensor=network.pin_map(config.sensor_port,"B1",4)
        ethernet=network.pin_map("LAN1","PLC",8)
        supply=network.connected("SUPPLY_L.conductor","IO_L.conductor")
        return_path=network.connected("SUPPLY_M.conductor","IO_M.conductor")
        short=network.connected("SUPPLY_L.conductor","SUPPLY_M.conductor") or network.connected("IO_L.conductor","IO_M.conductor")
        correct=sensor["correct"] and ethernet["correct"] and supply and return_path and not short
        circuit_held=self.circuit_held.update(correct,now);isolated=self.isolated_held.update(not short,now)
        pairs={tuple(sorted((e.a,e.b))) for e in network.contacts}
        bearing=set()
        if self.blade_port:
            for x,y in env.active_contact_pairs(shape_groups=(self.blade_mask,self.ferrule_mask)):
                blade,ferrule=(x,y) if x in self.blade_port else (y,x)
                bearing.add((self.blade_port[blade],int(self.shape_body[ferrule])))
        connections={}
        for port,cable,kind in self.destinations:
            raw,detail=self._landed(env,q,pairs,port,cable,kind,bearing)
            connections[port]=dict(cable=cable,raw_landed=raw,landed=self.held[port].update(raw,now),**detail)
        protected=self.integrity.update(env)
        restored=self.uplink_held.update(protected["conducting"],now)
        if self.geometry_tick is None or env.tick-self.geometry_tick>=20:
            self.routing=inspect_routing(env.assembly,q,net_traversal=True,bore_clearance=True)
            self.optical=self.commissioning.optics.read()
            self.geometry_tick=env.tick
        commissioning=self.commissioning.update(env,correct,short,optical=self.optical,
                                                sustained_loss=not (circuit_held and isolated))
        routed=all(x["passed_through"] for x in self.routing.values()) and len(self.routing)==4
        wiring=all(c["landed"] for c in connections.values()) and circuit_held and isolated
        touched,held_tools=self._robot_contact(env)
        speed=float(np.linalg.norm(env.read_array(env.state.body_qd)[self.moving,:3],axis=1).max(initial=0.))
        connector_rate=self.connector_motion.update(q[self.connector_ids,:3])
        cable_rate=self.motion.update(q[self.moving,:3])
        ready=wiring and routed and protected["continuity_preserved"] and commissioning["passed"]
        if ready and not touched and connector_rate is not None and connector_rate<MOTION_LIMIT_M_S:
            if self.since is None:self.since=env.tick
        else:self.since=None
        seconds=0. if self.since is None else (env.tick-self.since+1)*config.dt
        gates=dict(wiring=wiring,routing=routed,protected=protected["continuity_preserved"],
            commissioning=commissioning["passed"],hands_off_stable=seconds>=DWELL_S-config.dt/2)
        success=all(gates.values())
        units={port:int(c["landed"] and (isolated or c["cable"] not in SUPPLY_CABLES)) for port,c in connections.items()}
        units.update({guide:int(r["passed_through"]) for guide,r in self.routing.items()})
        units["PASS"]=int(commissioning["passed"])
        engagement={}
        for port,cable,kind in self.destinations:
            if connections[port]["landed"] or (cable in SUPPLY_CABLES and not isolated):continue
            depth=max(insertion_fraction(q,env.assembly.hardware[f"{cable}_end_{end}"]["body"],kind,self.frames[port])
                      for end in (0,1))
            if depth>0:engagement[port]=PARTIAL_ENGAGEMENT_UNIT*depth
        fraction=(sum(units.values())+sum(engagement.values()))/self.unit_count
        factor=1. if not protected["violation_latched"] else (VIOLATION_FACTOR if restored else UNRESTORED_VIOLATION_FACTOR)
        score=1. if success else UNIT_FACTOR*fraction*factor
        return dict(success=success,score=score,task_revision=2,
            qualification="assisted_digital_evidence_scoped_in_dossier",gates=gates,
            progress_units=units,engagement_progress=engagement,progress_fraction=fraction,
            uplink_conducting_debounced=restored,
            score_definition="1.0 if success else 0.9*(units + 0.5*insertion depth of each unlanded scheduled end in its own destination)/13, "
                             "x0.5 after a latched LAN2 violation if the uplink conducts again, x0.25 if it is still out",
            connections=connections,sensor=sensor,network=ethernet,supply=supply,return_path=return_path,
            supply_short=short,circuit_correct=correct,circuit_correct_debounced=circuit_held,
            supply_isolated_debounced=isolated,debounce_s=DEBOUNCE_S,
            robot_contact=touched,robot_held_tools=[env.builder.body_label[i] for i in held_tools],
            max_cable_speed_m_s=speed,hands_off_seconds=seconds,
            settled_motion_rate_m_s=connector_rate,settled_motion_bodies="eight new connector bodies",
            cable_motion_rate_m_s=cable_rate,motion_window_s=self.connector_motion.seconds,
            settled_motion_definition="Maximum per-body position-envelope diagonal over every physics step in the trailing window, divided by window seconds",
            settled_motion_limit_m_s=MOTION_LIMIT_M_S,
            routing=self.routing,protected=protected,commissioning=commissioning,
            geometry_evaluation_tick=self.geometry_tick,geometry_sample_interval_ticks=20)


def progress_for(env):
    return (ServiceProgress if env.config.task_revision>=2 else Progress)(env)

"""Existing uplink: a genuinely latched plug, flexible lead and potted far end."""
import numpy as np
import warp as wp
from scipy.interpolate import CubicSpline
from .electrical import Edge

def install(a):
    b=a.builder
    socket=np.asarray(a.hardware["LAN2"]["frame"])
    frame=wp.transform_multiply(wp.transform(socket[:3],socket[3:]),
                                wp.transform((0,0,-.017),wp.quat_identity()))
    plug=a.rj45_plug("protected_uplink",frame)
    start=np.asarray(wp.transform_point(frame,wp.vec3(0,0,.0327)))
    way=np.array([start,[.478,.140,1.155],[.475,.215,1.240],
                  [.515,.230,1.300],[.576,.230,1.300]])
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(way,axis=0),axis=1))]
    spline=CubicSpline(s,way,bc_type=((1,[-1,0,0]),(1,[1,0,0])))
    dense=spline(np.linspace(0,s[-1],2001))
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
    length=float(arc[-1]);segments=32;h=length/segments
    points=np.column_stack([np.interp(np.linspace(0,length,segments+1),arc,dense[:,k]) for k in range(3)])
    first_shape=b.shape_count
    bodies,joints=b.add_rod(points.tolist(),radius=.00265,body_frame_origin="com",
        label="protected_uplink_cable",cfg=a.shape_config(mu=.45,density=1200),
        stretch_stiffness=100000/h,shear_stiffness=100000/h,
        bend_stiffness=.00070/h,twist_stiffness=.00030/h,
        stretch_damping=.05,shear_damping=.05,bend_damping=2e-5/h,twist_damping=1e-5/h)
    a.cable_bodies.extend(bodies);a.cable_joints.extend(joints)
    for body in bodies:a._mass(body,.036*h)
    for shape in range(first_shape,b.shape_count):a.materials[shape]="cable_blue"
    a.objects["protected_uplink_cable"]=bodies
    a.cable_inventory["protected_uplink"]=dict(length_m=length,diameter_m=.0053,
        line_mass_kg_m=.036,bending_rigidity_nm2=.00070,torsional_rigidity_nm2=.00030,
        segments=segments,kind="rj45_potted_uplink",protected=True)
    world=wp.transform(start,wp.transform_get_rotation(frame))
    parent=wp.transform((0,0,.0327),wp.quat_identity())
    child=wp.transform_multiply(wp.transform_inverse(b.body_q[bodies[0]]),world)
    j=b.add_joint_fixed(plug["body"],bodies[0],parent_xform=parent,child_xform=child,
        collision_filter_parent=True,label="protected_uplink_manufactured_strain_relief")
    b.add_articulation([j],label="protected_uplink_bond");a.cable_joints.append(j)
    # Only the factory-potted far end is permanently fixed. The LAN2 plug
    # has NO world constraint, retention joint, servo or assistance.
    j=b.add_joint_fixed(-1,bodies[-1],parent_xform=b.body_q[bodies[-1]],
        collision_filter_parent=False,label="existing_uplink_potted_far_end")
    b.add_articulation([j],label="existing_potting");a.cable_joints.append(j)
    a.box("uplink_gateway",(.602,.230,1.300),(.024,.044,.025),material="steel")
    a.label("UPLINK / KEEP CONNECTED",(.575,.230,1.342),.110,.009)
    for i in range(1,9):
        a.fixed_edges.append(Edge(f"protected_uplink.{i}",f"UPLINK.{i}",1.724e-8*length/.205e-6))
    a.protected_connections=[dict(port="LAN2",destination="UPLINK",pins=8,
        connector="protected_uplink",cable="protected_uplink")]

class Integrity:
    """Latch a sustained continuity loss; initial contact settling is explicit."""
    def __init__(self):
        self.failed=False;self.good_seen=False;self.gap_since=None
    def update(self,env):
        values={x["port"]:env.network.pin_map(x["port"],x["destination"],x["pins"])
                for x in env.assembly.protected_connections}
        conducting=all(x["correct"] for x in values.values())
        now=env.tick*env.config.dt
        if conducting:self.good_seen=True;self.gap_since=None
        elif self.gap_since is None:self.gap_since=now
        # The initially authored seated interfaces may settle for 250 ms.
        # Thereafter a 20 ms persistent opening latches a task failure.
        if now>=.250 and not conducting and now-self.gap_since>=.020:self.failed=True
        return dict(conducting=conducting,ever_conducted=self.good_seen,
            preserved=conducting and not self.failed,violation_latched=self.failed,
            continuity_preserved=self.good_seen and not self.failed,
            pending_dropout_s=0. if self.gap_since is None else now-self.gap_since,
            initial_settling_window_s=.250,dropout_debounce_s=.020,circuits=values)

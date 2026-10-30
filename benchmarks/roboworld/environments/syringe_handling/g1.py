"""Supported-pelvis Unitree G1 / Dex3 deployment, using the shared pinned model.

A visible rigid rehabilitation stand holds the pelvis; this is explicitly assisted
upper-body manipulation, not free balancing or walking. All 43 URDF joints remain
physical PD actuators. No Franka arms or parallel-finger emulation are substituted.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation

PELVIS=(0.,-.28,.79)
ARMS_DOWN={f'g1/{side}_{joint}_joint':v for side in ('left','right') for joint,v in
           [('shoulder_pitch',-.1),('shoulder_roll',.3 if side=='left' else -.3),('elbow',1.3)]}

def setup_scene(sc,setup):
    import robots
    from robots.backends.newton import mount
    from newton._src.solvers.mujoco.equality import _add_equality_constraint
    first=sc.b.body_count;shape=sc.b.shape_count
    sc.rig_pending=robots.add_to(sc.b,setup,mounts={'g1_floor_mount':mount(PELVIS,math.pi/2)})
    sc.robot_bodies=list(range(first,sc.b.body_count))
    handle=sc.rig_pending.instances['g1'];sc.joints.update(handle.joints);sc.arms['g1']=handle
    for i in range(shape,sc.b.shape_count):sc.materials[i]='robot'
    pelvis=next(i for i in sc.robot_bodies if sc.b.body_label[i].endswith('/pelvis'))
    _add_equality_constraint(sc.b,constraint_type=sc.newton.solvers.SolverMuJoCo.EqType.WELD,
        body1=-1,body2=pelvis,relpose=sc.wp.transform(PELVIS, sc.wp.quat_from_axis_angle(sc.wp.vec3(0,0,1),math.pi/2)),enabled=True,label='g1_pelvis_support',
        custom_attributes={'mujoco:eq_solref':(.005,1.),'mujoco:eq_solimp':(.99,.999,.001,.5,2.)})
    # Stand is behind the robot; its connecting yoke is visual only, avoiding
    # redundant collisions with the rigid support constraint.
    sc.box('g1_support_base',(0,-.57,.035),(.25,.23,.035),'cart_frame',place='robot_support')
    sc.box('g1_support_post',(0,-.57,.44),(.035,.035,.40),'cart_frame',place='robot_support')
    sc.box('g1_support_yoke',(0,-.44,.79),(.08,.13,.025),'cart_frame',collide=False)

def setup_world(w,m):
    w._arm={};w._finger_speed={};w._arm_hand={};w._arm_fingers={};w._tcp_off={}
    w.grip_force={};w.grasp_latched={};w._grip_ref={};w.grip_applied={};w._finger_kp={}
    w.wrench_cmd={};w.load_kg={};w.load_cfg={};w._prev_cvel=None
    w._arm_mj_dofs={'g1':[]};w._g1_v=np.zeros(len(w.joints))
    for name,s in w.joints.items():
        j=w.mj_joint_of_dof[s['dof']];mdof=int(m.jnt_dofadr[j]);w._arm_mj_dofs['g1'].append((name,mdof))
        m.jnt_actfrclimited[j]=True;m.jnt_actfrcrange[j]=(-s['effort_limit'],s['effort_limit'])

def ready(w):
    q=w.model.joint_q.numpy()
    for n,v in ARMS_DOWN.items():q[w.joints[n]['q']]=v
    for s in w.joints.values():w.tq[s['q']]=w.targets[s['q']]=q[s['q']]
    w.model.joint_q.assign(q);w.state.joint_q.assign(q)
    w.newton.eval_fk(w.model,w.model.joint_q,w.model.joint_qd,w.state)

def servo(w):
    # Velocity/acceleration-limited reference, PD gains and torque limits from
    # the reusable barcode_scanning G1 profile. These are radian hand joints.
    for i,(name,s) in enumerate(w.joints.items()):
        q,df=s['q'],s['dof'];error=w.targets[q]-w.tq[q];a=8. if '_hand_' in name else 4.
        vmax=min(s['velocity_limit'],1.);v=np.sign(error)*min(vmax,math.sqrt(2*a*abs(error)))
        w._g1_v[i]+=np.clip(v-w._g1_v[i],-a*w.dt,a*w.dt)
        step=w._g1_v[i]*w.dt
        if abs(step)>abs(error):step=error;w._g1_v[i]=0.
        w.tq[q]+=step;w.tqd[df]=w._g1_v[i]

def check(w):
    for n,s in w.joints.items():
        speed=abs(w.jqd[s['dof']])
        if speed>1.2*s['velocity_limit']:
            raise FloatingPointError(f'G1 velocity limit: {n} {speed:g} rad/s')

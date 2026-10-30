"""Selected physical robot chain; IK produces servo commands, never pose writes."""
import numpy as np
from scipy.spatial.transform import Rotation
from robots.controllers.kinematics import SerialKinematics,transform

class Kinematics:
    def __init__(self,assembly):
        self.controllers={}
        for name,handle in assembly.builder._gar_robots.items():
            self.controllers[name]=SerialKinematics(handle.metadata['urdf'],'wsg_base',
                base=transform(assembly.robot_bases[name],(0,0,assembly.robot_yaws[name])),tip_offset=(0,0,.142))
        # Compatibility for older, explicitly FR3-only fixtures.
        first=next(iter(self.controllers.values()))
        self.chain=first.chain;self.lower=first.lower;self.upper=first.upper
    def fk(self,arm,q,**options):return self.controllers[arm].fk(q,**options)
    def solve(self,arm,position,orientation_xyzw,seed,**options):
        return self.controllers[arm].solve(position,orientation_xyzw,seed,**options)
    def joint_names(self,arm):
        return [arm+'/'+name for name in self.controllers[arm].names]

def arm_q(sim,arm):
    handle=sim.builder._gar_robots[arm];q=sim.state.joint_q.numpy()
    return np.array([q[handle.joints[name]['q']] for name in handle.groups['arm']['joints']])

def physical_tcp(sim,arm,ik=None):
    handle=sim.builder._gar_robots[arm];pose=handle.frame_pose('tcp',sim.state)
    q=pose['orientation_wxyz'];t=np.eye(4)
    t[:3,:3]=Rotation.from_quat([*q[1:],q[0]]).as_matrix();t[:3,3]=pose['position_m']
    return t

def validate_fk(sim):
    ik=Kinematics(sim.assembly);result={}
    for arm in sim.assembly.robot_bases:
        expected=physical_tcp(sim,arm,ik);actual=ik.fk(arm,arm_q(sim,arm))
        result[arm]=dict(position_error_m=float(np.linalg.norm(expected[:3,3]-actual[:3,3])),
            orientation_error_rad=float((Rotation.from_matrix(expected[:3,:3])*Rotation.from_matrix(actual[:3,:3]).inv()).magnitude()),
            tcp_m=actual[:3,3].tolist())
    if any(x['position_error_m']>2e-6 or x['orientation_error_rad']>2e-6 for x in result.values()):
        raise AssertionError('Kinematic source/world-frame mismatch: '+str(result))
    return result

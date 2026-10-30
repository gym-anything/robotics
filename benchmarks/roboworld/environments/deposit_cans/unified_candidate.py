"""Unified robot/shell base used by the corrected runtime revisions.

Native joint/contact equations and existing shell correction share one VBD
iteration loop. Implicit motor drives are explicitly effort-bounded. Gravity
compensation is joint feedforward INSIDE those bounds, not unlimited body lift.
No virtual robot proxies, lagged reaction-force feedback, object pose writes or
hidden grasp constraints. Requires separate robot/contact/dynamics qualification.
"""
import numpy as np
import newton
import warp as wp
from .bounded_drive import install

@wp.kernel
def gravity_feedforward(joints:wp.array[int],descendants:wp.array2d[int],
    parents:wp.array[int],dof_start:wp.array[int],kind:wp.array[int],
    joint_local:wp.array[wp.transform],axes:wp.array[wp.vec3],
    q:wp.array[wp.transform],com:wp.array[wp.vec3],mass:wp.array[float],
    gravity:wp.vec3,feedforward:wp.array[float]):
    i=wp.tid();joint=joints[i];parent=parents[joint];dof=dof_start[joint]
    frame=joint_local[joint]
    if parent>=0:frame=q[parent]*frame
    origin=wp.transform_get_translation(frame)
    axis=wp.normalize(wp.transform_vector(frame,axes[dof]));value=float(0.)
    for j in range(descendants.shape[1]):
        if descendants[i,j]:
            force=mass[j]*gravity
            if kind[joint]==int(newton.JointType.REVOLUTE):
                value-=wp.dot(axis,wp.cross(wp.transform_point(q[j],com[j])-origin,force))
            else:value-=wp.dot(axis,force)
    feedforward[dof]=value

class MotorController:
    def __init__(self,sim,solver):
        self.sim=sim;self.solver=solver;self.generated_sources=install(solver)
        m=sim.model;parents=m.joint_parent.numpy();children=m.joint_child.numpy()
        kinds=m.joint_type.numpy();starts=m.joint_qd_start.numpy()
        by_dof={int(starts[i]):i for i in range(len(kinds))
            if kinds[i]in(int(newton.JointType.REVOLUTE),int(newton.JointType.PRISMATIC))}
        joints=[by_dof[s['dof']]for s in sim.assembly.robot_joints.values()]
        ancestry={int(c):int(p)for p,c in zip(parents,children)}
        descendants=np.zeros((len(joints),m.body_count),np.int32)
        for i,joint in enumerate(joints):
            root=int(children[joint])
            for body in range(m.body_count):
                cursor=body;seen=set()
                while cursor>=0:
                    if cursor in seen:raise ValueError('Joint graph contains a cycle')
                    seen.add(cursor)
                    if cursor==root:descendants[i,body]=1;break
                    cursor=ancestry.get(cursor,-1)
        self.joints=wp.array(joints,dtype=int,device=m.device)
        self.descendants=wp.array(descendants,dtype=int,device=m.device)
    def prepare(self,state):
        m=self.sim.model
        self.solver.motor_feedforward.zero_()
        for name in ('motor_evaluation','passive_evaluation','limit_evaluation'):
            getattr(self.solver,name).zero_()
        wp.launch(gravity_feedforward,len(self.joints),[self.joints,self.descendants,
            m.joint_parent,m.joint_qd_start,m.joint_type,m.joint_X_p,m.joint_axis,
            state.body_q,m.body_com,m.body_mass,wp.vec3(0.,0.,-9.80665),
            self.solver.motor_feedforward],device=m.device)

def make(sim):
    # Structural-rest geometry must agree with the authored joint coordinates,
    # per Newton's basic joint example. This precedes ALL physical steps.
    newton.eval_fk(sim.model,sim.model.joint_q,sim.model.joint_qd,sim.model)
    from .solver import Coupled
    return Coupled(sim,monolithic=True)

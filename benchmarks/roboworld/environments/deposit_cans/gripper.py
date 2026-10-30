"""Explicit WSG MOVE/GRIP approximation using the native bounded motor.

Force is TOTAL gripper force, half per jaw. GRIP uses a speed-governed drive,
not an object attachment or a claim of measured holding. A later finger-width
command restores MOVE. Model arrays are the authoritative persistent mode.
"""
import math
import numpy as np
import newton

DRIVE_ARRAYS=('joint_target_ke','joint_target_kd','joint_target_mode','joint_effort_limit')


def validate(e,commands):
    if commands is None:return {}
    if not isinstance(commands,dict):raise ValueError('Grip commands must map arm to total force')
    solver=e.solver.native if hasattr(e.solver,'graphs')else e.solver
    if commands and solver.coupler is not None:
        raise ValueError('Explicit GRIP requires the integrated bounded native robot drive')
    result={}
    for arm,force in commands.items():
        if arm not in ('left','right'):raise ValueError('Unknown WSG arm')
        if isinstance(force,bool) or not isinstance(force,(int,float)) or not math.isfinite(force) or not 5<=force<=80:
            raise ValueError('WSG total gripping force must be in [5,80]N')
        if e.config.gripper_speed<.025:raise ValueError('GRIP requires the declared25mm/s per-jaw speed')
        result[arm]=float(force)
    return result


def apply(e,targets,commands):
    # Called only after the entire action has been validated, under env.lock.
    moving={n.split('/')[0]for n in targets if '/wsg_finger_joint' in n}
    if moving & set(commands):raise ValueError('Choose width MOVE or force GRIP per arm, not both')
    if not moving and not commands:return
    values={n:getattr(e.model,n).numpy()for n in DRIVE_ARRAYS}
    velocity=e.control.joint_target_qd.numpy()
    for arm in moving|set(commands):
        ids=[e.joints[f'{arm}/wsg_finger_joint{i}']['dof']for i in (1,2)]
        grip=arm in commands
        force=commands[arm]if grip else e.config.gripper_force
        values['joint_effort_limit'][ids]=force/2
        values['joint_target_ke'][ids]=0. if grip else 1800.
        values['joint_target_kd'][ids]=(force/2)/.025 if grip else 8.
        values['joint_target_mode'][ids]=int(newton.JointTargetMode.VELOCITY if grip else newton.JointTargetMode.POSITION_VELOCITY)
        velocity[ids]=-.025 if grip else 0.
    for n,v in values.items():getattr(e.model,n).assign(v)
    e.control.joint_target_qd.assign(velocity)


def state(e):
    mode=e.model.joint_target_mode.numpy();effort=e.model.joint_effort_limit.numpy()
    velocity=e.control.joint_target_qd.numpy();result={}
    for arm in ('left','right'):
        ids=[e.joints[f'{arm}/wsg_finger_joint{i}']['dof']for i in (1,2)]
        result[arm]=dict(mode='grip'if np.all(mode[ids]==int(newton.JointTargetMode.VELOCITY))else'move',
            total_effort_limit_n=float(effort[ids].sum()),target_jaw_velocity_m_s=velocity[ids].tolist(),
            holding_detected=None)
    return result

"""Labelled assisted FR3 grasp/hold/lift/release, one uninterrupted rollout.

Initial folded cloth is placed between open fingers. During closure ONLY,
distributed positioning forces hold it while cancelling cloth weight. Then support is zero and physical
contacts must carry the cloth. No pin, weld, grasp flag or cloth pose update
occurs after tick zero. This is component qualification, not a solved task.
"""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy.spatial.transform import Rotation
from control_cable.kinematics import Kinematics
from .assembly import HOME,folded_grid
from .config import Config
from .simulation import Simulation


def kinematics(sim):
    k=Kinematics(sim.assembly)
    for arm,yaw in sim.assembly.robot_yaws.items():
        k.bases[arm][:3,:3]=Rotation.from_euler('z',yaw).as_matrix()
    return k


def arm_q(sim,arm):
    q=sim.state_0.joint_q.numpy()
    return np.array([q[sim.assembly.robot_joints[f'{arm}/fr3_joint{i}']['q']] for i in range(1,8)])


def set_arm(sim,arm,q):
    for i,value in enumerate(q):
        sim.targets[sim.assembly.robot_joints[f'{arm}/fr3_joint{i+1}']['q']]=value


def gripper(sim,arm,opening):
    for name,spec in sim.assembly.robot_joints.items():
        if name.startswith(arm+'/') and 'finger' in name:
            sim.targets[spec['q']]=opening/2


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--cloth-solver',choices=['style3d','vbd'],default='style3d')
    p.add_argument('--grip-depth',type=float,default=.015)
    p.add_argument('--friction-velocity',type=float,default=.01)
    p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--native-coupling',action='store_true')
    p.add_argument('--native-rigid-iterations',type=int,default=20)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    c=Config(cloth_count=1,cloth_solver=args.cloth_solver,friction_velocity=args.friction_velocity,
             nonlinear_iterations=args.iterations,coupling='native' if args.native_coupling else 'lagged',
             contact_capacity=65536 if args.native_coupling else 32768,native_rigid_iterations=args.native_rigid_iterations)
    sim=Simulation(c)
    ik=kinematics(sim)
    tcp=ik.fk('left',arm_q(sim,'left'))
    uv,points,tris=folded_grid(c,(0,0,0),True)
    if not 0<args.grip_depth<.045:raise ValueError('Fixture grip depth must stay below the hand housing')
    local=np.column_stack((points[:,0],points[:,2]-.004,points[:,1]+.2-args.grip_depth))
    initial=local@tcp[:3,:3].T+tcp[:3,3]
    for state in (sim.state_0,sim.state_1):
        state.particle_q.assign(initial.astype(np.float32));state.particle_qd.zero_()
    support=np.zeros_like(initial,np.float32)
    support[:,2]=sim.model.particle_mass.numpy()*9.80665
    records=[];particles=[];bodies=[];ticks=[];phases=[]
    phase='supported_closure'
    def record(s):
        if s.tick%20:return
        summary=s.summary()
        q=arm_q(s,'left');current_tcp=ik.fk('left',q)
        shape=s.contacts.soft_contact_shape.numpy()
        count=int(s.contacts.soft_contact_count.numpy()[0])
        forces=s.solver.collision.forces.numpy()[:count]
        b=s.assembly.builder
        selected=[i for i in range(count) if b.shape_body[shape[i]]>=0 and 'finger' in b.body_label[b.shape_body[shape[i]]]]
        jaw=s.state_0.joint_q.numpy()
        summary.update(phase=phase,tcp=current_tcp[:3,3].tolist(),finger_contacts=len(selected),
            finger_force_xyz=forces[selected].sum(0).tolist() if selected else [0.,0.,0.],
            sum_finger_force_n=float(np.linalg.norm(forces[selected],axis=1).sum()) if selected else 0.,
            jaw_positions={name:float(jaw[spec['q']]) for name,spec in s.assembly.robot_joints.items() if 'finger' in name})
        records.append(summary);particles.append(s.state_0.particle_q.numpy().copy())
        bodies.append(s.state_0.body_q.numpy().copy());ticks.append(s.tick);phases.append(phase)
    record(sim)
    start=time.perf_counter()
    try:
        gripper(sim,'left',0.)
        mass=sim.model.particle_mass.numpy()[:,None]
        for i in range(480):
            # Labelled distributed positioning fixture, not a permanent pin.
            # Critically damped 20 rad/s support vanishes completely at tick 480.
            force=support+mass*(400.*(initial-sim.state_0.particle_q.numpy())-40.*sim.state_0.particle_qd.numpy())
            sim.step(1,fixture_forces=force.astype(np.float32),callback=record)
        phase='unsupported_hold';sim.step(300,callback=record)
        phase='physical_lift';start_pose=ik.fk('left',arm_q(sim,'left'))
        seed=arm_q(sim,'left');quat=Rotation.from_matrix(start_pose[:3,:3]).as_quat()
        for i in range(75):
            pos=start_pose[:3,3]+np.array([0,0,.05*(i+1)/75])
            target=ik.solve('left',pos,quat,seed)
            if not target['converged']:raise RuntimeError(f'Fixture IK failed: {target}')
            seed=np.asarray(target['joints']);set_arm(sim,'left',seed)
            sim.step(20,callback=record)
        phase='lifted_hold';sim.step(300,callback=record)
        phase='release';gripper(sim,'left',.08);sim.step(480,callback=record)
        phase='fall_after_release';sim.step(300,callback=record)
        error=None
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    elapsed=time.perf_counter()-start
    np.savez_compressed(out/'rollout.npz',particle_q=np.asarray(particles),body_q=np.asarray(bodies),
                        ticks=np.asarray(ticks),phases=np.asarray(phases))
    result=dict(config=asdict(c),fixture=__doc__,grip_depth_m=args.grip_depth,sources=sim.assembly.sources,error=error,
        wall_s=elapsed,sim_s=sim.tick*c.dt,wall_sim_ratio=elapsed/max(sim.tick*c.dt,1e-9),
        initial_tcp=tcp.tolist(),records=records)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),error=error,tick=sim.tick,wall_sim_ratio=result['wall_sim_ratio'])),flush=True)
    if error:raise RuntimeError(error)


if __name__=='__main__':main()

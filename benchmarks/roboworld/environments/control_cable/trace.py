"""Bounded, action-linked substep history for the mixed-solver workcell."""
from dataclasses import asdict
import shutil
from gym_anything_robotics.physics_trace import PhysicsTraceStore

class Trace:
    def __init__(self,env,path):
        self.env=env
        self.store=PhysicsTraceStore(path,max_bytes=256*1024**2,max_items=100_000,compression="zstd")
        self.current=None
        self.previous=None
        from .force_observer import RobotForces
        self.robot_forces=RobotForces(env)

    def begin(self,context):
        if self.current:raise RuntimeError("Trace already active")
        if shutil.disk_usage(self.store.path.parent).free<256*1024**2:
            raise OSError("Insufficient trace disk reserve; no action executed")
        command=str(context["action_id"])
        self.store.begin(command,str(context["request_id"]),self.env.tick,self.env.config.dt,
            {"source":"control-cable-newton15","generation":self.env.generation,
             "configuration":asdict(self.env.config),
             "body_labels":self.env.builder.body_label,"shape_labels":self.env.builder.shape_label,
             "sampling":"Every completed physics substep; no extra physics or collision refresh",
             "contact_forces":"Post-solve native VBD force-law values on shape1 in world axes; not measured impulses",
             "force_completeness":"Native VBD force-law output, native MuJoCo contact export and generalized actuator/constraint forces, and actually applied final-inner-iteration ADMM contact forces/body wrenches",
             "force_timing":"ADMM copied immediately after force accumulation, before dual updates; VBD/MuJoCo report final inner solve. Inner optimization iterations are not extra physical substeps.",
             "mujoco_wrench_reference":"World force/torque at the contact; native MuJoCo conversion does not add a COM moment arm. Not a measured impulse.",
             "geometry":"Use the matching privileged geometry catalog for body/shape identities"})
        self.current=command
        self.previous=set(tuple(sorted(p)) for p in self.env.contact_pairs())
        self.env.trace_callback=self.append
        return {"available":True,"id":command,"store_path":str(self.store.path)}

    def append(self,env):
        if shutil.disk_usage(self.store.path.parent).free<256*1024**2:
            raise OSError("Trace reserve reached after completed physics step")
        c=env.contacts
        read=env.read_array
        n=int(read(c.rigid_contact_count)[0])
        contacts={k:read(getattr(c,"rigid_contact_"+k))[:n] for k in
            ("shape0","shape1","point0","point1","offset0","offset1","normal","margin0","margin1")}
        pairs=set(tuple(sorted(p)) for p in env.contact_pairs())
        events=[]
        for kind,changed in (("contact_begin",pairs-self.previous),("contact_end",self.previous-pairs)):
            events.extend(dict(type=kind,tick=env.tick,shape_ids=list(p)) for p in sorted(changed))
        for kind,changed in env.electrical_events.items():
            events.extend(dict(type=kind,tick=env.tick,electrodes=list(p)) for p in changed)
        self.previous=pairs
        from .force_observer import vbd_contact_outputs
        vc,native,body_map=vbd_contact_outputs(env,env.previous_q)
        count=int(read(native[-1])[0])
        # Independent ordered contact table. Never zip these forces with the
        # parent contacts: native filtering changes its length AND row order.
        vbd=dict(shape0=read(vc.rigid_contact_shape0)[:count],
            shape1=read(vc.rigid_contact_shape1)[:count],
            force_world_on_shape1_n=read(native[4])[:count],
            point0_world_m=read(native[2])[:count],
            point1_world_m=read(native[3])[:count],
            normal_world=read(vc.rigid_contact_normal)[:count])
        sample=dict(tick=env.tick,generation=env.generation,contacts_step_start_tick=env.tick-1,
            **{k:read(getattr(env.state,k)) for k in ("body_q","body_qd","joint_q","joint_qd")},
            actuator_setpoints=read(env.control.joint_target_q),requested_targets=env.targets,
            contacts=contacts,vbd_contacts=vbd,
            admm_applied=env.solver.contact_observation(reader=env.read_array,prefetch=env.prefetch_arrays),
            robot_forces=self.robot_forces.read(),
            electrical_contacts=[asdict(e) for e in env.network.contacts],evaluation=env.evaluation)
        self.store.append(self.current,env.tick,sample,events)

    def finish(self):
        if self.current is None:return None
        command,self.current=self.current,None
        self.env.trace_callback=None
        self.store.finish(command,self.env.tick,status="failed" if self.env.fault else "completed",error=self.env.fault)
        return self.store.summary(command)

    def close(self):
        self.finish()
        self.store.close()

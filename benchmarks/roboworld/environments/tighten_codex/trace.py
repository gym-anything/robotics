"""Bounded, privileged every-step physics history for GAR actions."""

import shutil
import numpy as np
from gym_anything_robotics.physics_trace import PhysicsTraceStore


class Trace:
    def __init__(self,env,path):
        self.env=env
        # Every tick is committed; WAL with synchronous=NORMAL survives an
        # application crash without an fsync per tick (the measured cost).
        self.store=PhysicsTraceStore(path,max_bytes=512*1024**2,
                                     max_items=100_000,compression="zstd",synchronous="NORMAL")
        self.current=None
        self.previous=None

    def begin(self,context):
        if self.current is not None:
            raise RuntimeError("A trace is active")
        if shutil.disk_usage(self.store.path.parent).free<512*1024**2:
            self.env.fault="Insufficient trace disk reserve"
            raise OSError(self.env.fault)
        command=str(context["action_id"])
        try:
            self.store.begin(command,str(context["request_id"]),self.env.tick,
                self.env.config.dt,{"source":"tighten-codex-newton15",
                    "generation":self.env.generation,
                    "sampling":"Every physics tick, post-integration state; contact witnesses are step-start",
                    "force_impulses":"Solver contact force times dt is step impulse",
                    "mechanism":"Native helical equality, solved head/bearing contact, load-dependent friction and clamp retention"})
        except Exception as exc:
            self.env.fault=f"Trace could not begin: {exc}"
            raise
        self.current=command
        self.previous=None
        self.env.trace_callback=self.append
        return {"available":True,"id":command,"store_path":str(self.store.path)}

    def append(self,env):
        # Step-start candidate pairs are kept as shape ids (contact begin/end
        # events stay exact). Witness geometry, force and couple are kept for
        # every force-bearing solved contact; zero-force candidates (mostly
        # FR3 link-to-link proximity inside the 25 um margin) carry no load
        # and their geometry is not stored. The environment has already read
        # back this tick's solved contacts for its load evaluation.
        c=env.contacts
        n=int(c.rigid_contact_count.numpy()[0])
        shape0=c.rigid_contact_shape0.numpy()[:n]
        shape1=c.rigid_contact_shape1.numpy()[:n]
        solved=env.solved_contacts
        count=int(solved.rigid_contact_count.numpy()[0])
        force=solved.force.numpy()[:count]
        loaded=np.flatnonzero(np.abs(force).max(axis=1)>0) if count else np.zeros(0,np.int64)
        pairs={tuple(sorted((int(a),int(b)))) for a,b in zip(shape0,shape1)}
        events=[]
        if self.previous is not None:
            for kind,changed in (("contact_begin",pairs-self.previous),
                                 ("contact_end",self.previous-pairs)):
                events.extend({"type":kind,"tick":env.tick,"shape_ids":list(pair)}
                              for pair in sorted(changed))
        self.previous=pairs
        if shutil.disk_usage(self.store.path.parent).free<512*1024**2:
            raise OSError("Trace disk reserve reached")
        sample={"tick":env.tick,"generation":env.generation,
            "contacts_step_start_tick":env.tick-1,
            **{key:getattr(env.state,key).numpy()
               for key in ("body_q","body_qd","joint_q","joint_qd")},
            "actuator_setpoints":env.control.joint_target_q.numpy(),
            "requested_targets":env.targets,
            "contacts":{"shape0":shape0,"shape1":shape1,"candidate_count":n},
            "solved_contacts":{
                "solved_count":count,"loaded_index":loaded,
                **{key:getattr(solved,"rigid_contact_"+key).numpy()[:count][loaded]
                   for key in ("shape0","shape1","point0","point1")},
                "force_world_on_shape0_n":force[loaded,:3],
                "contact_torque_world_on_shape0_nm":force[loaded,3:],
                "impulse_world_on_shape0_n_s":force[loaded,:3]*env.config.dt},
            "evaluation":env.evaluation}
        self.store.append(self.current,env.tick,sample,events)

    def retention(self):
        """Eviction status of the bounded store, for every action result."""
        with self.store._read_snapshot():
            evicted=self.store.db.execute(
                "SELECT COALESCE(SUM(evicted_samples),0),COALESCE(SUM(evicted_events),0) FROM trace").fetchone()
            oldest=self.store.db.execute(
                "SELECT MIN(tick) FROM item WHERE kind='sample'").fetchone()[0]
        return {"evicted_samples_total":int(evicted[0]),"evicted_events_total":int(evicted[1]),
                "oldest_retained_sample_tick":oldest,
                "budget":{"max_payload_bytes":512*1024**2,"max_items":100_000}}

    def finish(self):
        if self.current is None:return None
        command,self.current=self.current,None
        self.env.trace_callback=None
        self.store.finish(command,self.env.tick,
                          status="failed" if self.env.fault else "completed",
                          error=self.env.fault)
        return self.store.summary(command)

    def close(self):
        self.finish()
        self.store.close()

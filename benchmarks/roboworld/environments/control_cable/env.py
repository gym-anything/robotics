"""Persistent workcell; public commands address robot servos only."""
from dataclasses import asdict
import math
import threading
import numpy as np
import warp as wp
import newton
from .physics import Physics
from .scene import Workcell
from .config import Config
from .electrical import Network, metal_contact_edges
from .task import progress_for

def development_config(**changes):
    """Explicit candidate settings; component qualification is still in progress."""
    from dataclasses import replace
    c=Config(dt=1/3840,convex_hardware=True,contact_hard=False,rigid_relaxation=.5,
        joint_angular_penalty=100,terminal_anchor_y=.0031,terminal_inward_blade=True,
        terminal_blade_lead_m=.0008,m12_negative_entry_chamfer=True,
        m12_contact_entry_lead_m=.0006,m12_spring_contacts=True,
        contact_pose_warmstart=True,friction_static_tangent_fix=True,
        friction_epsilon_m_s=.001,cable_linear_hard=True,
        cable_constraint_penalty_n_m=100000.,rigid_force_space=True,
        rigid_contact_cone="elliptic",coupling_rho=2000.)
    return replace(c,**changes)

class ControlCableEnv(Physics):
    def __init__(self,config=None,*,assembly=None,diagnostic_passive_only=False):
        if assembly is not None and (not isinstance(assembly,Workcell) or assembly.config!=config):
            raise ValueError("Fixture assembly must be the same workcell and exact configuration")
        super().__init__(assembly if assembly is not None else Workcell(config or development_config()),
                         passive_only=diagnostic_passive_only)
        self.newton=newton
        self.lock=threading.RLock()
        self.joints=self.assembly.robot_joints
        # Shape ownership is immutable for this model; episode replacement
        # constructs a new environment and therefore a new ownership table.
        self._shape_body=np.asarray(self.builder.shape_body,dtype=np.int32)
        self.generation=0
        self.renderer=None
        self.trace_callback=None
        self._reset_observers()
        self.enable_graphs()

    def _reset_observers(self):
        self.network=Network(self.assembly.fixed_edges)
        self.progress=progress_for(self)
        self.evaluation=self.progress.initial()
        self.protective_stops=[]
        self.previous_q=wp.clone(self.state.body_q)
        self.electrical_events={}

    def _step(self,*,fixture_body_forces=None):
        wp.copy(self.previous_q,self.state.body_q)
        super().step(fixture_body_forces=fixture_body_forces)
        self._step_read_cache=dict(self._completed_host_state)
        self._vbd_observation=None
        try:
            arrays=([getattr(self.contacts,"rigid_contact_"+name) for name in
                ("shape0","shape1","point0","point1","normal","margin0","margin1")]+
                ([self.contacts.rigid_contact_offset0,self.contacts.rigid_contact_offset1] if self.trace_callback else []))
            if self.trace_callback:
                # These ADMM snapshots were produced during the completed step.
                # Batch their copies with the raw contacts instead of a second
                # GPU synchronization later in the trace callback.
                arrays+=self.solver.observation_arrays()
            self.prefetch_arrays(arrays)
            edges=metal_contact_edges(self,self.previous_q)
            self.electrical_events=self.network.update(edges)
            self.evaluation=self.progress.update(self)
            if self.trace_callback:self.trace_callback(self)
        except Exception as exc:
            # Physics already advanced: report its actual tick and stop.
            self.fault=f"Observation/recording failed after tick {self.tick}: {type(exc).__name__}: {exc}"
            raise
        finally:
            self._step_read_cache=None
            self._vbd_observation=None

    def validate_targets(self,targets,ticks):
        if type(ticks) is not int or ticks<=0:raise ValueError("Positive integer physics ticks required")
        pending=self.targets.copy()
        for name,value in targets.items():
            if name not in self.joints:raise ValueError(f"Not a robot actuator: {name}")
            spec=self.joints[name]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f"Finite target required: {name}")
            if not spec["lower"]<=value<=spec["upper"]:raise ValueError(f"Target outside manufacturer limits: {name}")
            pending[spec["q"]]=value
        return pending

    def act(self,targets,*,ticks,guard=None):
        with self.lock:
            if self.fault:raise RuntimeError(self.fault)
            pending=self.validate_targets(targets,ticks)
            self.targets=pending
            for _ in range(ticks):
                self._step()
                stop=self._hardware_protective_stop()
                if stop:
                    # The shared workcell stop brakes both arms at their measured
                    # postures; neither keeps driving an inaccessible target.
                    self.fr3_trajectory.stop_all(
                        self.state, self.targets, self.control
                    )
                    stop["scope"]="all_arms"
                    self.protective_stops.append(stop)
                    break
                if guard and guard(self):break
            return self.get_state()

    def _hardware_protective_stop(self):
        for a,b in self.active_contact_pairs():
            bodies=[self._shape_body[i] for i in (a,b)]
            names=[self.model.body_label[i] if i>=0 else "world" for i in bodies]
            arms=[n.split("/")[0] if n.startswith(("left/","right/")) and "finger" not in n else None
                  for n in names]
            if arms[0] is not None and bodies[0]!=bodies[1]:
                return dict(tick=self.tick,arm=arms[0],bodies=names,
                    shapes=[self.builder.shape_label[a],self.builder.shape_label[b]])
            if arms[1] is not None and bodies[0]!=bodies[1]:
                return dict(tick=self.tick,arm=arms[1],bodies=names,
                    shapes=[self.builder.shape_label[a],self.builder.shape_label[b]])
        velocities=self.state.joint_qd.numpy()
        for name,spec in self.joints.items():
            if "finger" in name:
                continue
            speed=abs(float(velocities[spec["dof"]]))
            limit=float(spec["velocity_limit"])
            if speed>1.05*limit:
                return dict(tick=self.tick,arm=name.split("/")[0],kind="velocity",
                    joint=name,speed_rad_s=speed,limit_rad_s=limit)
        return None

    def contact_pairs(self):
        n=int(self.read_array(self.contacts.rigid_contact_count)[0])
        a=self.read_array(self.contacts.rigid_contact_shape0)[:n]
        b=self.read_array(self.contacts.rigid_contact_shape1)[:n]
        return [(int(x),int(y)) for x,y in zip(a,b)]

    def get_state(self):
        return self._get_state()

    def active_contact_pairs(self,*,shape_groups=None):
        from scipy.spatial.transform import Rotation
        n=int(self.read_array(self.contacts.rigid_contact_count)[0])
        a=self.read_array(self.contacts.rigid_contact_shape0)[:n]
        b=self.read_array(self.contacts.rigid_contact_shape1)[:n]
        selected=np.arange(n)
        if shape_groups is not None:
            left,right=shape_groups
            selected=selected[(left[a]&right[b])|(right[a]&left[b])]
        if not len(selected):return []
        q=self.read_array(self.state.body_q)
        world=[]
        for side in (0,1):
            p=self.read_array(getattr(self.contacts,"rigid_contact_point"+str(side)))[selected].copy()
            # Raw support witnesses: margins below already include effective radii.
            ids=self._shape_body[(a if side==0 else b)[selected]]
            moving=ids>=0
            p[moving]=q[ids[moving],:3]+Rotation.from_quat(q[ids[moving],3:]).apply(p[moving])
            world.append(p)
        distance=np.sum((world[1]-world[0])*self.read_array(self.contacts.rigid_contact_normal)[selected],axis=1)
        for side in (0,1):distance-=self.read_array(getattr(self.contacts,"rigid_contact_margin"+str(side)))[selected]
        active=selected[distance<=0]
        return [(int(x),int(y)) for x,y in zip(a[active],b[active])]

    def _get_state(self):
        with self.lock:
            q,v=self.state.body_q.numpy(),self.state.body_qd.numpy()
            jq,jv=self.state.joint_q.numpy(),self.state.joint_qd.numpy()
            return dict(tick=self.tick,generation=self.generation,time_s=self.tick*self.config.dt,
                ready=self.fault is None,fault=self.fault,config=asdict(self.config),
                joints={n:dict(position=float(jq[s["q"]]),velocity=float(jv[s["dof"]]),
                    target=float(self.targets[s["q"]])) for n,s in self.joints.items()},
                bodies={n:dict(id=i,position_m=q[i,:3].tolist(),quaternion_xyzw=q[i,3:].tolist(),
                    velocity_linear_angular=v[i].tolist()) for i,n in enumerate(self.model.body_label)},
                hardware=dict(protective_stops=list(self.protective_stops)),
                electrical_contacts=[asdict(e) for e in self.network.contacts],
                electrical_events=self.electrical_events,evaluation=self.evaluation,
                contact_count=int(self.contacts.rigid_contact_count.numpy()[0]))

    def reset(self):
        with self.lock:
            if self.trace_callback:raise RuntimeError("Finish the active trace before reset")
            # Rebuild solver internals too: pose-only assignment is not a reset.
            Physics.__init__(self,self.assembly)
            self._reset_observers()
            self.generation+=1
            self.enable_graphs()
            return self.get_state()

    def close(self):
        with self.lock:
            if self.renderer:
                self.renderer.close()
                self.renderer=None

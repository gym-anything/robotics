"""Read-only instrumentation of the last ADMM inner solve, pinned to Newton 1.5.

Copies go to separate observer buffers. No force, multiplier, control, state,
contact manifold or iteration count is changed. Captures the force inputs
BEFORE the subsequent dual update, not a post-step reconstruction.
"""
import numpy as np
import warp as wp
import newton
from newton.solvers.experimental.coupled import SolverCoupledADMM
from newton._src.solvers.coupled.admm_utils import contact_surface_point

@wp.kernel
def snapshot(active:wp.array[int],ba:wp.array[int],bb:wp.array[int],
             pa:wp.array[wp.vec3],oa:wp.array[wp.vec3],
             pb:wp.array[wp.vec3],ob:wp.array[wp.vec3],
             qa:wp.array[wp.transform],qb:wp.array[wp.transform],
             rho:float,W:wp.array[float],lam:wp.array[wp.vec3],
             u:wp.array[wp.vec3],jv:wp.array[wp.vec3],
             force:wp.array[wp.vec3],xa:wp.array[wp.vec3],xb:wp.array[wp.vec3]):
    i=wp.tid()
    if i<active[0]:
        force[i]=W[i]*(lam[i]+rho*W[i]*(u[i]-jv[i]))
        xa[i]=contact_surface_point(qa[ba[i]],pa[i],oa[i])
        xb[i]=contact_surface_point(qb[bb[i]],pb[i],ob[i])

class RecordedADMM(SolverCoupledADMM):
    def __init__(self,*args,**kwargs):
        self.observer_buffers={}
        super().__init__(*args,**kwargs)
        self.observer_body_wrenches={name:wp.zeros_like(buf.body_f) for name,buf in self._admm_buffers.items() if buf.body_f is not None}
        self.observer_external_wrenches={name:wp.zeros_like(value) for name,value in self.observer_body_wrenches.items()}
        self._ensure_observers()

    def _ensure_observers(self):
        for g in self._admm_dynamic_rr_contact_groups:
            if id(g) not in self.observer_buffers:
                self.observer_buffers[id(g)]=tuple(wp.zeros(g.count,dtype=wp.vec3,device=self.model.device) for _ in range(3))

    def prepare_contacts(self,*args,**kwargs):
        value=super().prepare_contacts(*args,**kwargs)
        self._ensure_observers()
        return value

    def _accumulate_admm_forces(self,iteration_k,dt,*,refresh_jv,initialize_contact_u):
        for name,target in self.observer_external_wrenches.items():
            wp.copy(target,self._admm_buffers[name].body_f)
        super()._accumulate_admm_forces(iteration_k,dt,refresh_jv=refresh_jv,
                                       initialize_contact_u=initialize_contact_u)
        for g in self._admm_dynamic_rr_contact_groups:
            if not g.count:continue
            a=self._entries[g.body_entry_name_a];b=self._entries[g.body_entry_name_b]
            wp.launch(snapshot,dim=g.count,inputs=[g.active_count,g.body_ids_a,g.body_ids_b,
                g.point_a,g.offset_a,g.point_b,g.offset_b,a.state_0.body_q,b.state_0.body_q,
                float(self._coupling.rho),g.W,g.lambda_,g.u,g.Jv],
                outputs=list(self.observer_buffers[id(g)]),device=self.model.device)
        for name,target in self.observer_body_wrenches.items():
            wp.copy(target,self._admm_buffers[name].body_f)

    def observation_arrays(self):
        """Already-completed observer buffers; enumeration performs no reads/writes."""
        arrays=[]
        for g in self._admm_dynamic_rr_contact_groups:
            a=self._entries[g.body_entry_name_a];b=self._entries[g.body_entry_name_b]
            arrays.extend([g.active_count,g.shape_ids_a,g.shape_ids_b,
                a.body_local_to_global,b.body_local_to_global,g.body_ids_a,g.body_ids_b,
                g.normal,g.friction,*self.observer_buffers[id(g)]])
        for name,value in self.observer_body_wrenches.items():
            arrays.extend([self._entries[name].body_local_to_global,value,
                self.observer_external_wrenches[name]])
        return arrays

    def contact_observation(self,*,reader=None,prefetch=None):
        read=reader or (lambda array:array.numpy())
        if prefetch:prefetch(self.observation_arrays())
        result=[]
        for g in self._admm_dynamic_rr_contact_groups:
            count=int(read(g.active_count)[0])
            if not 0<=count<=g.count:raise OverflowError("ADMM observer contact capacity exceeded")
            a=self._entries[g.body_entry_name_a];b=self._entries[g.body_entry_name_b]
            force,xa,xb=self.observer_buffers[id(g)]
            result.append(dict(entry_a=a.name,entry_b=b.name,count=count,
                shape_a=read(g.shape_ids_a)[:count],shape_b=read(g.shape_ids_b)[:count],
                body_a=read(a.body_local_to_global)[read(g.body_ids_a)[:count]],
                body_b=read(b.body_local_to_global)[read(g.body_ids_b)[:count]],
                point_a_world_m=read(xa)[:count],point_b_world_m=read(xb)[:count],
                force_world_on_a_n=read(force)[:count],
                normal_world=read(g.normal)[:count],friction=read(g.friction)[:count]))
        return dict(contacts=result,applied_body_wrenches={name:dict(
            body_ids=read(self._entries[name].body_local_to_global),
            total_world_wrench_n_nm=read(value),
            pre_contact_world_wrench_n_nm=read(self.observer_external_wrenches[name])) for name,value in self.observer_body_wrenches.items()})

def vbd_contact_outputs(physics,previous_q):
    """Use the SAME local pose, contact ordering and history as the native solve.

    Coupled views compact bodies (but retain global shape IDs), and filter /
    reorder contacts. Passing global poses or the parent contact list to an
    entry's collect_rigid_contact_forces is incorrect. Read the existing entry
    buffer directly: entry_contacts() can run filtering and is not an observer.
    The history selection matches Newton 1.5's own VBD proxy-force harvesting.
    """
    if getattr(physics,"_step_read_cache",None) is not None and getattr(physics,"_vbd_observation",None) is not None:
        return physics._vbd_observation
    coupled=physics.solver;solver=coupled.solver("cables")
    if hasattr(coupled,"_entries"):
        entry=coupled._entries["cables"]
        contacts=coupled._entry_contact_buffers.get("cables")
        if contacts is None:raise RuntimeError("No solved cable contact buffer")
        q=entry.state_1.body_q
        previous=solver._coupling_body_q_prev_snapshot
        body_map=entry.body_local_to_global
    else:
        contacts=physics.contacts;q=physics.state.body_q;previous=previous_q
        body_map=None
    native=solver.collect_rigid_contact_forces(q,previous,contacts,physics.config.dt)
    result=contacts,native,body_map
    if getattr(physics,"_step_read_cache",None) is not None:
        physics._vbd_observation=result
        physics.prefetch_arrays([contacts.rigid_contact_shape0,contacts.rigid_contact_shape1,
            contacts.rigid_contact_normal,native[4],native[-1]]+
            ([native[2],native[3]] if getattr(physics,"trace_callback",None) else []))
    return result


class RobotForces:
    """Native MuJoCo contact export into OWN buffers; never refreshes collision."""
    def __init__(self,env):
        self.env=env;self.entry=env.solver._entries["rigid"];self.solver=self.entry.solver
        self.contacts=newton.Contacts(self.solver.get_max_contact_count(),0,
            requested_attributes={"force"},device=env.model.device)
        self.shape_map=self.entry.attribute_local_to_global[newton.Model.AttributeFrequency.SHAPE].numpy()
        local=self.solver.mjc_dof_to_newton_dof.numpy()
        self.dof_map=np.full(local.shape,-1,dtype=int)
        valid=local>=0
        self.dof_map[valid]=self.entry.joint_dof_local_to_global.numpy()[local[valid]]

    def read(self):
        solver=self.solver;c=self.contacts;solver.update_contacts(c)
        read=self.env.read_array;data=solver.mjw_data
        self.env.prefetch_arrays([c.rigid_contact_count,c.rigid_contact_shape0,c.rigid_contact_shape1,
            c.rigid_contact_point0,c.rigid_contact_point1,c.force,
            data.qfrc_actuator,data.qfrc_constraint,data.actuator_force])
        n=int(read(c.rigid_contact_count)[0])
        if not 0<=n<=c.rigid_contact_max:raise OverflowError("MuJoCo observer contact capacity exceeded")
        shapes=[read(getattr(c,"rigid_contact_shape"+str(i)))[:n] for i in (0,1)]
        if any(np.any(s<0) for s in shapes):raise ValueError("Unmapped MuJoCo contact geometry")
        data=solver.mjw_data
        return dict(count=n,shape0=self.shape_map[shapes[0]],shape1=self.shape_map[shapes[1]],
            point0_body_m=read(c.rigid_contact_point0)[:n],
            point1_body_m=read(c.rigid_contact_point1)[:n],
            native_contact_wrench_world_on_shape0=read(c.force)[:n],
            # These arrays retain the MuJoCo DOF indexing; explicit robot model
            # mapping avoids pretending it equals the complete world DOF array.
            qfrc_actuator=read(data.qfrc_actuator),qfrc_constraint=read(data.qfrc_constraint),
            actuator_force=read(data.actuator_force),
            mujoco_dof_to_global_newton_dof=self.dof_map,
            actuator_names=[solver.mj_model.actuator(i).name for i in range(solver.mj_model.nu)])

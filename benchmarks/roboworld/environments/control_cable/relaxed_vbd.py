"""Under-relaxed native Newton 1.5 rigid VBD sweep (diagnostic until qualified).

The unmodified full Newton update oscillates across the contact active set on
the small terminal/ferrule. Relax each colour's COM/rotation increment BEFORE
the native dual update. No forces, masses, geometry or constitutive terms change.
This rigid-only stepping adapter follows Newton v1.5.0 SolverVBD's
_solve_rigid_body_iteration, NVIDIA/Newton Developers, Apache-2.0.
Source: https://github.com/newton-physics/newton/blob/v1.5.0/newton/_src/solvers/vbd/solver_vbd.py
"""
import warp as wp
from newton.solvers import SolverVBD
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    accumulate_body_body_contacts_per_body, solve_rigid_body,
    update_duals_body_body_contacts, update_duals_joint,
    _NUM_CONTACT_THREADS_PER_BODY)

@wp.kernel
def warmstart_contact_pose(counts:wp.array[int],inverse_mass:wp.array[float],
                          previous:wp.array[wp.transform],q:wp.array[wp.transform]):
    b=wp.tid()
    if inverse_mass[b]>0.0 and counts[b]>0:
        q[b]=previous[b]

@wp.kernel
def relax_color(ids:wp.array[int],old:wp.array[wp.transform],
                com:wp.array[wp.vec3],factor:float,q:wp.array[wp.transform]):
    b=ids[wp.tid()]
    x0=wp.transform_point(old[b],com[b]);x1=wp.transform_point(q[b],com[b])
    r0=wp.transform_get_rotation(old[b]);r1=wp.transform_get_rotation(q[b])
    # Native solve_rigid_body normalizes rot_new. Warp 1.16 slerp multiplies
    # q0 by an axis-angle quaternion WITHOUT normalizing its result. Repeated
    # interpolation therefore preserved/amplified q0's floating-point norm
    # drift, undoing Newton's unit-rotation invariant. Normalizing this
    # rotational update is representation maintenance, not a pose clamp.
    r=wp.normalize(wp.quat_slerp(wp.normalize(r0),wp.normalize(r1),factor))
    x=x0+factor*(x1-x0)
    q[b]=wp.transform(x-wp.quat_rotate(r,com[b]),r)

class RelaxedVBD(SolverVBD):
    def _initialize_rigid_bodies(self,state_in,control,contacts,dt,refresh):
        super()._initialize_rigid_bodies(state_in,control,contacts,dt,refresh)
        if self.contact_pose_warmstart and contacts is not None:
            # Change the initial nonlinear iterate, NOT the inertial objective,
            # completed state, force law, contact geometry or velocity history.
            wp.launch(warmstart_contact_pose,dim=self.model.body_count,
                inputs=[self.body_body_contact_counts,self.body_inv_mass_effective,
                    self.body_q_prev],outputs=[state_in.body_q],device=self.device)

    def __init__(self,model,*,relaxation=.5,contact_pose_warmstart=False,**kwargs):
        if model.particle_count:raise ValueError("This pinned adapter supports rigid rods, not particles")
        if not 0<relaxation<=1:raise ValueError("Relaxation must be in (0,1]")
        super().__init__(model,**kwargs)
        self.relaxation=relaxation
        self.contact_pose_warmstart=contact_pose_warmstart
        self._before_color=wp.clone(model.body_q)
        # Newton 1.5 builds an empty 1-D joint_dof_dim for a jointless
        # model, while the native rigid kernel requires a 2-D array.
        # Supply an empty correctly typed view; never add a dummy constraint.
        self._joint_dof_dim=(model.joint_dof_dim if model.joint_count else
            wp.empty((0,2),dtype=int,device=self.device))

    def _solve_rigid_body_iteration(self,state_in,state_out,control,contacts,dt):
        m=self.model
        if self.integrate_with_external_rigid_solver or not m.body_count:return
        self.body_torques.zero_();self.body_forces.zero_()
        self.body_hessian_aa.zero_();self.body_hessian_al.zero_();self.body_hessian_ll.zero_()
        for group in m.body_color_groups:
            if contacts is not None:
                wp.launch(accumulate_body_body_contacts_per_body,
                    dim=group.size*_NUM_CONTACT_THREADS_PER_BODY,
                    inputs=[dt,group,self.body_q_prev,state_in.body_q,m.body_com,
                        self.body_inv_mass_effective,self.friction_epsilon,
                        self.body_body_contact_penalty_k,self.body_body_contact_material_ke,
                        self.body_body_contact_material_kd,self.body_body_contact_material_mu,
                        self.body_body_contact_lambda,self.body_body_contact_C0,
                        self.rigid_contact_alpha,self.rigid_contact_hard,
                        contacts.rigid_contact_count,contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,contacts.rigid_contact_offset0,
                        contacts.rigid_contact_offset1,contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,contacts.rigid_contact_margin1,
                        m.shape_body,self.body_body_contact_buffer_pre_alloc,
                        self.body_body_contact_counts,self.body_body_contact_indices],
                    outputs=[self.body_forces,self.body_torques,self.body_hessian_ll,
                        self.body_hessian_al,self.body_hessian_aa],device=self.device)
            wp.copy(self._before_color,state_in.body_q)
            wp.launch(solve_rigid_body,dim=group.size,inputs=[
                dt,group,state_in.body_q,self.body_q_prev,m.body_q,m.body_mass,
                self.body_inv_mass_effective,m.body_inertia,self.body_inertia_q,
                m.body_com,self.rigid_adjacency,m.joint_type,m.joint_enabled,
                m.joint_parent,m.joint_child,m.joint_X_p,m.joint_X_c,m.joint_axis,
                self.joint_cable_rest_kb_local,self.joint_cable_rest_twist,
                m.joint_qd_start,m.joint_target_q_start,self.joint_constraint_start,
                self.joint_penalty_k,self.joint_penalty_kd,self.joint_sigma_start,
                self.joint_C_fric,m.joint_target_ke,m.joint_target_kd,
                control.joint_target_q,control.joint_target_qd,m.joint_limit_lower,
                m.joint_limit_upper,m.joint_limit_ke,m.joint_limit_kd,
                self.joint_lambda_lin,self.joint_lambda_ang,self.joint_C0_lin,
                self.joint_C0_ang,self.joint_is_hard,self.rigid_joint_alpha,
                self._joint_dof_dim,self.joint_rest_angle,self.body_forces,
                self.body_torques,self.body_hessian_ll,self.body_hessian_al,
                self.body_hessian_aa],outputs=[state_in.body_q],device=self.device)
            wp.launch(relax_color,dim=group.size,
                inputs=[group,self._before_color,m.body_com,self.relaxation],
                outputs=[state_in.body_q],device=self.device)
        if contacts is not None:
            wp.launch(update_duals_body_body_contacts,dim=contacts.rigid_contact_max,
                inputs=[contacts.rigid_contact_count,contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,contacts.rigid_contact_offset0,
                    contacts.rigid_contact_offset1,contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,contacts.rigid_contact_margin1,
                    m.shape_body,state_in.body_q,self.body_q_prev,
                    self.body_body_contact_material_mu,self.body_body_contact_C0,
                    self.rigid_contact_alpha,self.rigid_contact_hard,
                    self.body_body_contact_material_ke,self.rigid_linear_beta,
                    self.body_body_contact_penalty_k,self.body_body_contact_lambda],device=self.device)
        if m.joint_count:
            wp.launch(update_duals_joint,dim=m.joint_count,inputs=[
                m.joint_type,m.joint_enabled,m.joint_parent,m.joint_child,
                m.joint_X_p,m.joint_X_c,m.joint_axis,self.joint_cable_rest_kb_local,
                self.joint_cable_rest_twist,m.joint_qd_start,m.joint_target_q_start,
                self.joint_constraint_start,state_in.body_q,m.body_q,m.joint_dof_dim,
                self.joint_C0_lin,self.joint_C0_ang,self.joint_is_hard,
                self.rigid_joint_alpha,self.joint_penalty_k_max,self.rigid_linear_beta,
                self.rigid_angular_beta,m.joint_target_ke,control.joint_target_q,
                m.joint_limit_lower,m.joint_limit_upper,m.joint_limit_ke,
                self.joint_rest_angle,self.joint_penalty_k,self.joint_lambda_lin,
                self.joint_lambda_ang],device=self.device)

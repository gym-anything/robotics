"""Diagnostic single-solver wrapper; never selected by the public environment."""
from newton.solvers import SolverVBD

def cable_solver(model,config):
    solver=SolverVBD;kwargs={}
    if config.rigid_relaxation!=1. or config.contact_pose_warmstart:
        from .relaxed_vbd import RelaxedVBD
        solver=RelaxedVBD;kwargs["relaxation"]=config.rigid_relaxation
        kwargs["contact_pose_warmstart"]=config.contact_pose_warmstart
    result=solver(model,**kwargs,iterations=config.vbd_iterations,
        friction_epsilon=config.friction_epsilon_m_s,
        rigid_joint_angular_ke=config.joint_angular_penalty,
        rigid_contact_hard=config.contact_hard,
        rigid_body_contact_buffer_size=config.body_contact_capacity,rigid_contact_history=False)
    if config.cable_linear_hard:
        import newton,numpy as np,warnings
        # Explicit alternative constitutive approximation: axial/shear rigid,
        # finite EI/GJ unchanged. Native AL rows replace penalty-only extension.
        # This is NOT silently described as the original finite-EA model.
        kinds=model.joint_type.numpy();starts=result.joint_constraint_start.numpy()
        hard=result.joint_is_hard.numpy()
        slots=[]
        penalty=float(config.cable_constraint_penalty_n_m)
        if not np.isfinite(penalty) or penalty<=0:raise ValueError("Positive numerical cable constraint penalty required")
        for j in np.flatnonzero(kinds==int(newton.JointType.CABLE)):
            hard[starts[j]:starts[j]+2]=1
            slots.extend((starts[j],starts[j]+1))
        result.joint_is_hard.assign(hard)
        # In hard mode the penalty is a numerical AL conditioning scale, not
        # the finite material EA/h. Use the native structural linear scale;
        # retained multipliers enforce inextensibility. EI/GJ stay finite.
        for name in ("joint_penalty_k","joint_penalty_k_min","joint_penalty_k_max"):
            values=getattr(result,name).numpy();values[slots]=penalty
            getattr(result,name).assign(values)
        warnings.warn(f"Inextensible cable numerical AL penalty: {penalty:g} N/m",stacklevel=2)
        warnings.warn("Explicit inextensible/unshearable cable experiment; finite bending/twist retained",stacklevel=2)
    return result


class SingleVBD:
    def __init__(self,model,config):
        self.inner=cable_solver(model,config)
    def solver(self,name):
        if name!="cables":raise KeyError(name)
        return self.inner
    def prepare_contacts(self,contacts):pass
    def step(self,*args,**kwargs):return self.inner.step(*args,**kwargs)
    def notify_model_changed(self,*args):return self.inner.notify_model_changed(*args)

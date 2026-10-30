"""Opt-in VBD paper section3.7 adaptive INITIAL GUESS, unchanged inertia.

Chen et al. 2024, https://arxiv.org/html/2403.06321v4#S3.SS7
The external acceleration remains in the implicit-Euler inertial target.
Only the iterative starting guess interpolates ballistic/constant-velocity
prediction using the previous actual acceleration along the external load.
No final position/velocity is clamped, frozen or prescribed; all contact,
elastic, gravity and finite-effort equations are solved as before.
"""
import warp as wp
from newton import ParticleFlags

@wp.kernel
def adaptive_forward(dt:float,gravity:wp.array[wp.vec3],world:wp.array[int],
                     pos_prev:wp.array[wp.vec3],pos:wp.array[wp.vec3],vel:wp.array[wp.vec3],
                     inv_mass:wp.array[float],external_force:wp.array[wp.vec3],flags:wp.array[int],
                     old_velocity:wp.array[wp.vec3],valid:wp.array[int],
                     inertia:wp.array[wp.vec3],displacements:wp.array[wp.vec3],factors:wp.array[float]):
    i=wp.tid();pos_prev[i]=pos[i]
    previous=old_velocity[i];old_velocity[i]=vel[i]
    if not flags[i]&ParticleFlags.ACTIVE or inv_mass[i]==0.:
        inertia[i]=pos[i];displacements[i]=wp.vec3(0.);factors[i]=0.;valid[i]=1;return
    acc=gravity[world[i]]+external_force[i]*inv_mass[i]
    predicted_velocity=vel[i]+acc*dt
    # Preserve native expression/order for byte-exact inertial targets.
    inertia[i]=pos[i]+predicted_velocity*dt
    alpha=float(1.);norm2=wp.dot(acc,acc)
    if valid[i]!=0 and norm2>1.e-20:
        observed=(vel[i]-previous)/dt
        alpha=wp.clamp(wp.dot(observed,acc)/norm2,0.,1.)
    displacements[i]=(vel[i]+alpha*acc*dt)*dt
    factors[i]=alpha;valid[i]=1

def initialize(solver,state_in,state_out,dt):
    m=solver.model
    if not m.particle_count:return
    if solver.particle_enable_self_contact:
        solver._collision_detection_penetration_free(state_in)
    else:
        solver.pos_prev_collision_detection.assign(state_in.particle_q)
        solver.particle_displacements.zero_()
    wp.launch(adaptive_forward,dim=m.particle_count,inputs=[dt,m.gravity,m.particle_world,
        solver.particle_q_prev,state_in.particle_q,state_in.particle_qd,m.particle_inv_mass,
        state_in.particle_f,m.particle_flags,solver.initial_guess_old_velocity,solver.initial_guess_valid],
        outputs=[solver.inertia,solver.particle_displacements,solver.initial_guess_factors],device=m.device)
    solver._penetration_free_truncation(state_in.particle_q)

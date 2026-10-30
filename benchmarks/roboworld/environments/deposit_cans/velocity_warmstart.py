"""Collision-bounded nonlinear initial guess; physical inertia is unchanged.

Use the last physical velocity to initialize the new position iterate, with
the existing simultaneous collision/discovery bound. Unlike the native inertial
predictor, this omits acceleration from the GUESS only; the complete gravity,
external force and implicit-inertia residual remain in the physical equations.
No velocity clipping, attachment, changed contact law or skipped check.
"""
from types import MethodType
import warp as wp
from deposit_cans.elastic_precision import candidate

@wp.kernel
def predict(dt:float,velocity:wp.array[wp.vec3],mass:wp.array[float],out:wp.array[wp.vec3]):
    i=wp.tid();out[i]=wp.vec3(0.)
    if mass[i]>0.:out[i]=dt*velocity[i]

def install(g):
    s=g.s;original=s._initialize_particles
    if not s.global_only or s.global_start!='previous' or g.precision is None:
        raise ValueError('Expected existing precise/global previous-position candidate')
    def initialize(self,state_in,state_out,dt):
        original(state_in,state_out,dt)
        m=self.model;p=g.precision
        wp.launch(predict,m.particle_count,[dt,state_in.particle_qd,m.particle_mass,g.delta],device=m.device)
        self.relative_step.increment(state_in.particle_q,g.delta,1.,g.displacement,g.trial)
        wp.launch(candidate,m.particle_count,[p.current,g.delta,self.relative_step.scale,1.,p.trial,g.trial,
            self.pos_prev_collision_detection,g.displacement],device=m.device)
        wp.copy(p.current,p.trial);wp.copy(state_in.particle_q,g.trial)
        wp.copy(state_out.particle_q,g.trial);wp.copy(self.particle_displacements,g.displacement)
    s._initialize_particles=MethodType(initialize,s)
    g.velocity_warmstart=True

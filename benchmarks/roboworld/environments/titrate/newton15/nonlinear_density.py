"""Sequential linearization of the end-of-step density constraint.

Solve for velocity u with q_trial=q_start+dt*u and rho(q_trial)<=rho0.
Each inner pressure projection minimizes change from the SAME unconstrained
velocity, not from the preceding trial. The linearization is
  rho(q_trial,b_start+dt*v_boundary)
                 + dt*G(q_trial)*(u_new-u_trial) <= rho0.
Rigid boundary transport retains the first-order partitioned discretization;
the nonlinear correction is in fluid displacement, not a coupled rigid solve.
Contact half-spaces stay at step start, like the ordinary time integrator.
Changing their closest triangle during this fluid-only SQP iteration makes
the constrained problem switch at faceted concave cavity corners.
Only a converged velocity is integrated once by the ordinary stepping loop.
Trial coordinates are scratch storage, never written to Newton physical state.
No positions are repaired, no timestep changed, and no velocity is clipped.

This repository-owned nonlinear extension needs separate physical acceptance.
It is not represented as the published first-order DFSPH algorithm.
The optional density-controlled variant stops on the true end density alone;
it is iterated density correction, not a fully stationary nonlinear minimum-
velocity-change solution. Its physical behaviour needs separate qualification.
"""
import numpy as np
import warp as wp
from .sph_kernels import prepare, pressure_grad, integrate, apply_velocity


@wp.kernel
def subtract_trial_rate(grid: wp.uint64, q: wp.array[wp.vec3], trial: wp.array[wp.vec3],
        volume: float, h: float, kind: int, bg: wp.array[wp.vec3], dt: float,
        rho: wp.array[float]):
    i=wp.tid()
    rate=wp.dot(trial[i],bg[i])
    for j in wp.hash_grid_query(grid,q[i],h):
        r=q[i]-q[j]
        if wp.length_sq(r)<h*h:
            rate+=volume*wp.dot(trial[i]-trial[j],pressure_grad(r,h,kind))
    # The inner projection separately subtracts boundary transport, so only
    # the fluid trial velocity belongs in this change of linearization origin.
    rho[i]-=dt*rate


class Workspace:
    def __init__(self,w):
        self.q=wp.empty_like(w.state.particle_q)
        self.trial=wp.empty_like(w.state.particle_qd)
        self.previous=wp.empty_like(self.trial)
        self.check_q=wp.empty_like(self.q)
        self.check_grid=wp.HashGrid(64,64,512,device=w.config.device)
        self.check_rho=wp.empty_like(w.rho);self.check_factor=wp.empty_like(w.factor)
        self.check_bg=wp.empty_like(w.bg);self.check_gf=wp.empty_like(w.bg)
        self.check_n=wp.empty_like(w.neighbors);self.check_rate=wp.empty_like(w.boundary_rate)
        self.line_velocity=wp.empty_like(self.trial)


@wp.kernel
def relax_trial(candidate: wp.array[wp.vec3], alpha: float, trial: wp.array[wp.vec3]):
    i=wp.tid()
    trial[i]+=alpha*(candidate[i]-trial[i])


def project(w,unconstrained,*,react=True):
    if w.ground:
        raise ValueError('Nonlinear density currently requires explicit solid boundaries or a free liquid fixture')
    if not hasattr(w,'nonlinear_workspace'):
        w.nonlinear_workspace=Workspace(w)
    s=w.nonlinear_workspace
    if w.boundaries is None:
        return _project(w,unconstrained,react=react)
    samples=w.boundaries.samples
    original_q,original_grid=samples.q,samples.grid
    if not hasattr(s,'boundary_q') or s.boundary_q.shape != original_q.shape:
        s.boundary_q=wp.empty_like(original_q)
        s.boundary_grid=wp.HashGrid(64,64,512,device=w.config.device)
    wp.launch(integrate,dim=len(original_q),inputs=[original_q,samples.v,w.config.dt_s],
        outputs=[s.boundary_q],device=w.config.device)
    s.boundary_grid.build(s.boundary_q,samples.coverage_support_m)
    # Only the quadrature used by this scratch solve is transported. Newton
    # rigid poses and fluid state stay untouched; restore the normal adapter
    # even when the nonlinear solve raises.
    samples.q,samples.grid=s.boundary_q,s.boundary_grid
    try:
        return _project(w,unconstrained,react=react)
    finally:
        samples.q,samples.grid=original_q,original_grid


def _project(w,unconstrained,*,react):
    s=w.nonlinear_workspace;cfg=w.config
    s.trial.assign(unconstrained)
    if w.boundaries is not None and cfg.nonlinear_linesearch:
        # Start the merit search inside its fixed contact half-spaces.
        # Subsequent candidates and their convex combinations then remain
        # contact-feasible; a density-only merit must not start with omitted
        # contact violations. This computes only a scratch velocity seed.
        w.dv.zero_()
        w.boundaries.contact_projection(w,unconstrained,w.active)
        w._launch(apply_velocity,[unconstrained,w.dv],[s.trial])
    start_q=w.state.particle_q
    w.constraint_positions=s.q
    def prepare_trial():
        w._launch(integrate,[start_q,s.trial,cfg.dt_s],[s.q])
        w.grid.build(s.q,cfg.support_m)
        if w.neighbor_cache is not None:w.neighbor_cache.update()
        w._launch(prepare,[w.grid.id,s.q,w.volume_per_particle_m3,cfg.support_m,0,w.kernel_kind,int(w.ordered_threads)],
            [w.rho,w.factor,w.bg,w.neighbors,w.fluid_gradient])
        if w.boundaries is not None:
            w.boundaries.prepare(w)
            # Boundary samples already occupy their predicted positions.
            # Subtracting boundary transport again would count it twice.
            w.boundary_rate.zero_()
    prepare_trial()
    records=[]
    reference=unconstrained.numpy().astype(np.float64)
    penalty=0.
    def check_velocity(velocity):
        w._launch(integrate,[start_q,velocity,cfg.dt_s],[s.check_q])
        s.check_grid.build(s.check_q,cfg.support_m)
        w._launch(prepare,[s.check_grid.id,s.check_q,w.volume_per_particle_m3,cfg.support_m,0,w.kernel_kind,int(w.ordered_threads)],
            [s.check_rho,s.check_factor,s.check_bg,s.check_n,s.check_gf])
        if w.boundaries is not None:
            from .sph_solid import prepare_boundary
            b=w.boundaries.samples
            w._launch(prepare_boundary,[s.check_gf,b.grid.id,s.check_q,b.q,b.v,
                b.spacing**3,w.volume_per_particle_m3,cfg.support_m,w.kernel_kind],
                [s.check_rho,s.check_factor,s.check_bg,s.check_rate,s.check_n])
        return s.check_rho.numpy()
    for iteration in range(cfg.nonlinear_iterations):
        initial_density=w.rho.numpy()
        w._launch(subtract_trial_rate,[w.grid.id,s.q,s.trial,w.volume_per_particle_m3,
            cfg.support_m,w.kernel_kind,w.bg,cfg.dt_s],[w.rho])
        s.previous.assign(s.trial)
        candidate=w._project(unconstrained,True,react=False)
        motions=np.linalg.norm(candidate.numpy()-s.previous.numpy(),axis=1)
        largest=int(np.argmax(motions))
        change=cfg.dt_s*float(motions[largest])
        end_density=check_velocity(candidate)
        density=float(np.maximum(end_density-1.,0.).max())
        records.append({'iteration':iteration+1,'maximum_density_error':density,
            'maximum_trial_motion_change_m':change,'maximum_trial_motion_particle':largest,
            'trial_relaxation':cfg.nonlinear_relaxation,
            'velocity_stationarity_required':cfg.nonlinear_stationarity})
        if density<=cfg.nonlinear_density_error and (not cfg.nonlinear_stationarity or change<=cfg.spacing_m*cfg.nonlinear_motion_error_fraction):
            w.last_projection['nonlinear']=records
            # The check used separate scratch arrays; accepted multipliers,
            # gradients and contact impulses still belong to the same solve.
            if w.boundaries is not None and react:w.boundaries.pressure_reaction(w)
            w.rho.assign(end_density)
            return candidate
        if cfg.nonlinear_linesearch:
            # Exact L1 density merit for SQP, in mass-normalized kinetic units.
            # p/dt is the density-constraint multiplier. This changes only
            # where the next linearization is evaluated: a damped trial is
            # never accepted as a physical step without full convergence.
            old=s.previous.numpy().astype(np.float64)
            direction=candidate.numpy().astype(np.float64)-old
            penalty=max(penalty,2.*float(w.p.numpy().max())/cfg.dt_s+1.)
            initial_excess=float(np.maximum(initial_density-1.,0.).sum(dtype=np.float64))
            initial_objective=.5*float(np.sum((old-reference)**2))
            slope=float(np.sum((old-reference)*direction))-penalty*initial_excess
            if slope>=0. and initial_excess>0.:
                penalty=max(penalty,2.*(float(np.sum((old-reference)*direction))+float(np.sum(direction**2)))/initial_excess)
                slope=float(np.sum((old-reference)*direction))-penalty*initial_excess
            initial_merit=initial_objective+penalty*initial_excess
            accepted=False
            for backtrack in range(9):
                alpha=2.**(-backtrack)
                value=old+alpha*direction
                if backtrack:
                    s.line_velocity.assign(value.astype(np.float32))
                    end_density=check_velocity(s.line_velocity)
                merit=.5*float(np.sum((value-reference)**2))+penalty*float(np.maximum(end_density-1.,0.).sum(dtype=np.float64))
                if merit<=initial_merit+1e-4*alpha*min(slope,0.):
                    s.trial.assign(value.astype(np.float32))
                    records[-1].update(linesearch_fraction=alpha,initial_merit=initial_merit,accepted_merit=merit)
                    accepted=True
                    break
            if not accepted:
                w.last_projection['nonlinear']=records
                raise FloatingPointError(f'Nonlinear density line search failed: merit {initial_merit:g} -> {merit:g}')
        else:
            # Relax the scratch linearization point, not the physical step.
            # Acceptance still checks the full pressure-corrected candidate
            # and the same density/motion tolerances. This damps alternating
            # SQP iterates without changing material forces or clipping motion.
            w._launch(relax_trial,[candidate,cfg.nonlinear_relaxation],[s.trial])
        prepare_trial()
    w.last_projection['nonlinear']=records
    raise FloatingPointError(f'Nonlinear density did not converge: {records[-1]}')

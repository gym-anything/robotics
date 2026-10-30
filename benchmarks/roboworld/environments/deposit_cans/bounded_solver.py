"""Explicit bounded-iteration candidate using the corrected v5 force field.

Uses the existing contact-coupled PSD search/preconditioner and line-work
checks, not a claim of exact nonlinear convergence. No residual is clamped,
no velocity is zeroed, and no physical coefficient or timestep is changed.
The strict reference remains selectable as residual-shell-v5.
"""
import numpy as np
import warp as wp
from .precise_self_geometry import install as self_geometry
from .body_point_precision import install as body_geometry
from .retention_precision import install as retention
from .residual_precision import ResidualPrecision
from .velocity_warmstart import install as warmstart


def install(coupled,config):
    g=coupled.native.global_refinement
    g.work_shortcuts=config.work_shortcuts
    g.batched_elastic_force=config.batched_elastic_force
    self_geometry(g);body_geometry(g);retention(coupled)
    field=g.accurate_residual=ResidualPrecision(g,full_contact=True)
    original=g.evaluate
    def evaluate(q,state,contacts,dt,*,hessians=True,contact_ready=False,precise_q=None):
        # Retain the existing search curvature, but assemble actual forces in
        # float64 before any summation is narrowed for public/native readback.
        if hessians:original(q,state,contacts,dt,hessians=True,contact_ready=contact_ready,precise_q=precise_q)
        field.evaluate(g.precision.current if precise_q is None else precise_q,state,contacts,dt)
    g.evaluate=evaluate
    prepare=g.linear.prepare_rhs
    def prepare_rhs():
        prepare()
        # Do not round the assembled residual to float32 and widen it again.
        wp.copy(g.linear.rhs,field.result)
    g.linear.prepare_rhs=prepare_rhs
    warmstart(g)
    g.bounded_endpoint_force=wp.zeros_like(field.result)
    refine=g.refine
    def refinement(state,out,contacts,dt):
        refine(state,out,contacts,dt)
        # Before velocity/history/plasticity finalization. Copied on every
        # refinement so the final copy is correct under CUDA graph replay too.
        wp.copy(g.bounded_endpoint_force,field.result)
    g.refine=refinement
    coupled.bounded_solver=Status(coupled,config)
    return coupled


class Status:
    def __init__(self,coupled,config):self.c=coupled;self.config=config
    def status(self):
        g=self.c.native.global_refinement
        f=g.bounded_endpoint_force.numpy();mass=self.c.model.particle_mass.numpy()
        if not np.isfinite(f).all():raise FloatingPointError('Nonfinite bounded endpoint residual')
        free=mass>0.;net=f[free].sum(0)
        return dict(revision='bounded-shell-v6',admission='fixed_iteration_budget_with_existing_finite_geometry_contact_guards',
            nonlinear_converged=None,maximum_vertex_residual_n=float(np.linalg.norm(f[free],axis=1).max()),
            all_free_vertex_net_residual_n=net.tolist(),native_sweeps=self.config.iterations,
            search='coupled_PSD_PCG_and_checked_line_work',force_accumulation='float64',
            velocity_initial_guess='previous_physical_velocity_collision_bounded',
            scope='Residual before plastic update; zero buffer at construction is not convergence. Budget completion is not a rest or accuracy pass.',
            qualification='Opt-in numerical candidate; physical and performance qualification pending')

"""Integrated residual-shell-v5 stepping, not an experiment launcher.

No task motion, initial-state fixture, replay, parameter sweep or hidden force.
A step either passes its final physical-force gate or faults before publication.
The newly integrated revision has not been executed/qualified.
"""
import copy
import math
from types import MethodType
import numpy as np
import warp as wp
from .residual_newton import ResidualNewton
from .residual_precision import ResidualPrecision
from .precise_self_geometry import install as self_geometry
from .body_point_precision import install as body_geometry
from .retention_precision import install as retention
from .interface_balance import InterfaceBalance

REVISION='residual-shell-v5'
FORCE_TOLERANCE_N=1e-5
INTERFACE_FORCE_TOLERANCE_N=1e-5
INTERFACE_TORQUE_TOLERANCE_NM=1e-5
# Iteration changes, NOT limits on physical motion over a timestep.
RIGID_TRANSLATION_TOLERANCE_M=1e-7
RIGID_ROTATION_TOLERANCE_RAD=1e-6


def refinement_count(iterations,stride):
    if type(iterations)is not int or iterations<1 or type(stride)is not int or stride<1:
        raise ValueError('Positive exact iteration count and stride required')
    return len(range(0,iterations,stride))+int((iterations-1)%stride!=0)


class ResidualStepper:
    def __init__(self,coupled,config):
        if config.capture:raise ValueError('residual-shell-v5 requires capture=False; host sparse solve cannot be CUDA-captured')
        # Resolve the declared linear backend before a physical step, with no
        # silent fallback to a different algorithm or reduced workload.
        if config.residual_linear_solver=='pardiso':
            from pypardiso import PyPardisoSolver
        self.c=coupled;self.s=coupled.native;self.g=self.s.global_refinement
        if not self.s.global_only or self.g is None or self.g.precision is None:
            raise ValueError('Residual stepping requires the precise global-only shell path')
        self.config=config;self.original_step=coupled.step
        self.expected=refinement_count(self.s.iterations,self.s.global_stride)
        self.solver=None;self.records=[];self.active=False
        g=self.g
        g.regularized_search=g.trust_region_search=g.least_squares_search=False
        g.pseudo_transient_search=True
        self_geometry(g);body_geometry(g);retention(coupled)
        g.accurate_residual=ResidualPrecision(g,full_contact=True)
        g.analytic_derivatives=True
        g.residual_newton=None;g.residual_records=self.records;g.refinement_number=0
        self.interface=InterfaceBalance(coupled)
        self.control=None;self.previous_penalty=None;self.previous_body=None
        rigid_iteration=self.s._solve_rigid_body_iteration
        def rigid(owner,state,out,control,contacts,dt):
            penalty=owner.body_particle_contact_penalty_k
            if self.previous_penalty is None or self.previous_penalty.shape!=penalty.shape:
                self.previous_penalty=wp.empty_like(penalty)
            wp.copy(self.previous_penalty,penalty)
            if self.previous_body is None:self.previous_body=wp.empty_like(state.body_q)
            wp.copy(self.previous_body,state.body_q)
            return rigid_iteration(state,out,control,contacts,dt)
        self.s._solve_rigid_body_iteration=MethodType(rigid,self.s)
        g.refine=self.refine

    def rigid_update(self,state):
        if self.previous_body is None:raise RuntimeError('Missing rigid iteration reference')
        selected=self.interface.dynamic
        before=self.previous_body.numpy()[selected].astype(float)
        after=state.body_q.numpy()[selected].astype(float)
        if not len(before):return 0.,0.
        if not np.isfinite(before).all()or not np.isfinite(after).all():raise FloatingPointError('Nonfinite rigid iterate')
        a=before[:,3:];b=after[:,3:]
        na=np.linalg.norm(a,axis=1);nb=np.linalg.norm(b,axis=1)
        if np.any(abs(na-1.)>1e-5)or np.any(abs(nb-1.)>1e-5):
            raise FloatingPointError('Invalid rigid quaternion during coupled solve')
        a=a/na[:,None];b=b/nb[:,None]
        b=np.where((np.sum(a*b,axis=1)<0.)[:,None],-b,b)
        angle=4.*np.arctan2(np.linalg.norm(a-b,axis=1),np.linalg.norm(a+b,axis=1))
        return float(np.linalg.norm(after[:,:3]-before[:,:3],axis=1).max()),float(angle.max())

    def refine(self,state,out,contacts,dt):
        if not self.active:raise RuntimeError('Residual refinement outside an owned physical step')
        index=len(self.records)
        if index>=self.expected:raise RuntimeError('Unexpected extra shell refinement')
        final=index==self.expected-1
        g=self.g
        g.accurate_residual.component='all'
        if self.solver is None:
            self.solver=ResidualNewton(g,state,contacts,dt,audit_derivatives=False)
            g.residual_newton=self.solver
        rounds=[];corrections=0
        for coupling in range(self.config.residual_coupling_maxiter if final else 1):
            # Native AVBD changes the soft penalty AFTER its body sweep.
            # Include that force-law change, not only shell motion, when
            # measuring the reaction the next body sweep must reconcile.
            if final and self.previous_penalty is None:raise RuntimeError('Missing rigid force reference')
            before=self.interface.evaluate(g.precision.current,state,contacts,dt,
                penalty=self.previous_penalty)if final else None
            # Repeatedly restarting at sigma=1 rejected all ten one-correction
            # intermediate solves in current_v5_001. Carry ONLY the adaptive
            # scalar within this timestep. New force/J/factor/contact data are
            # rebuilt, and the new problem must pass its own trial checks.
            hint=self.solver.regularization if index>0 or coupling>0 else None
            self.solver.begin_problem(state,contacts,dt,search_shift=hint)
            result,x=self.solver.solve(max_iterations=self.config.residual_maxiter if final else 1,
                absolute_tolerance=FORCE_TOLERANCE_N,pardiso=self.config.residual_linear_solver=='pardiso',
                reuse=True,persistent=True,verbose=False)
            corrections+=len(result['iterations'])
            # Publish only accepted iterates to the work buffers. The physical
            # state/history/clock are finalized once, after this coupled solve.
            g.precision.current.assign(x);g.q.assign(x.astype(np.float32))
            wp.copy(state.particle_q,g.q);wp.copy(out.particle_q,g.q)
            self.s.particle_displacements.assign(x.astype(np.float32)-self.s.pos_prev_collision_detection.numpy())
            if not final:break
            after=self.interface.evaluate(g.precision.current,state,contacts,dt)
            df,dtau=self.interface.difference(before,after)
            dx,angle=self.rigid_update(state)
            rounds.append(dict(force_change_n=df,torque_change_nm=dtau,
                rigid_translation_update_m=dx,rigid_rotation_update_rad=angle,
                shell_converged=result['converged'],corrections=len(result['iterations'])))
            if not result['converged']:break
            if (df<=INTERFACE_FORCE_TOLERANCE_N and dtau<=INTERFACE_TORQUE_TOLERANCE_NM
                and dx<=RIGID_TRANSLATION_TOLERANCE_M and angle<=RIGID_ROTATION_TOLERANCE_RAD):break
            if coupling+1==self.config.residual_coupling_maxiter:break
            # Reconcile the changed load with the actual bounded robot/frame
            # solve, not an added force or a second physical timestep. Inertia,
            # previous positions, controls and plastic history stay fixed.
            self.s._solve_rigid_body_iteration(state,out,self.control,contacts,dt)
            self.s._implicit_retention.follow(state)
            self.s.relative_step.refresh_if_needed(state)
        self.records.append(dict(refinement=index,final=final,converged=result['converged'],
            corrections=corrections,final_maximum_vertex_N=result['final_maximum_vertex_N'],
            final_l2_N=result['final_l2_N'],coupling=rounds))
        g.refinement_number=len(self.records)
        if final and not result['converged']:
            raise FloatingPointError('Unconverged final shell force residual: '+str(result['final_maximum_vertex_N'])+' N')
        if final and (df>INTERFACE_FORCE_TOLERANCE_N or dtau>INTERFACE_TORQUE_TOLERANCE_NM
            or dx>RIGID_TRANSLATION_TOLERANCE_M or angle>RIGID_ROTATION_TOLERANCE_RAD):
            raise FloatingPointError('Unreconciled shell/rigid endpoint: '+str((df,dtau,dx,angle))+' (N,Nm,m,rad)')

    def step(self,state,out,control,contacts,dt):
        if self.active:raise RuntimeError('Reentrant residual step')
        self.records=[];self.g.residual_records=self.records;self.g.refinement_number=0
        if control is None:control=self.c.model.control(clone_variables=False)
        self.active=True;self.control=control
        try:
            result=self.original_step(state,out,control,contacts,dt)
            if len(self.records)!=self.expected or not self.records[-1]['final']:
                raise RuntimeError('Missing final shell convergence check; clock not advanced')
            if not self.status()['final_converged']:
                raise FloatingPointError('Unconverged coupled endpoint; clock not advanced')
            return result
        finally:
            self.active=False;self.control=None
            # No factor cache crosses a completed or failed physical step.
            # Work/state buffers are NOT rolled back on failure.
            if self.solver is not None:self.solver.close()

    def status(self):
        return dict(revision=REVISION,force_tolerance_n=FORCE_TOLERANCE_N,
            search='defect_checked_pseudo_transient',linear_solver=self.config.residual_linear_solver,
            expected_refinements=self.expected,refinements=copy.deepcopy(self.records),
            endpoint_coupling_budget=self.config.residual_coupling_maxiter,
            interface_force_tolerance_n=INTERFACE_FORCE_TOLERANCE_N,
            interface_torque_tolerance_nm=INTERFACE_TORQUE_TOLERANCE_NM,
            rigid_iteration_translation_tolerance_m=RIGID_TRANSLATION_TOLERANCE_M,
            rigid_iteration_rotation_tolerance_rad=RIGID_ROTATION_TOLERANCE_RAD,
            convergence_scope='Shell momentum residual and shell-induced rigid reaction change; not a measurement of the full rigid/joint residual',
            final_converged=bool(len(self.records)==self.expected and self.records[-1]['final']
                and self.records[-1]['converged'] and self.records[-1]['coupling']
                and self.records[-1]['coupling'][-1]['force_change_n']<=INTERFACE_FORCE_TOLERANCE_N
                and self.records[-1]['coupling'][-1]['torque_change_nm']<=INTERFACE_TORQUE_TOLERANCE_NM
                and self.records[-1]['coupling'][-1]['rigid_translation_update_m']<=RIGID_TRANSLATION_TOLERANCE_M
                and self.records[-1]['coupling'][-1]['rigid_rotation_update_rad']<=RIGID_ROTATION_TOLERANCE_RAD),
            qualification='Integrated source; physical and performance qualification pending')

    def checkpoint_state(self):
        if self.active:raise ValueError('Checkpoint requires a residual action boundary')
        return copy.deepcopy(self.records)

    def validate_checkpoint(self,records):
        if not isinstance(records,list)or len(records)not in (0,self.expected):
            raise ValueError('Incomplete checkpoint residual status')
        keys={'refinement','final','converged','corrections','final_maximum_vertex_N','final_l2_N','coupling'}
        for index,row in enumerate(records):
            if not isinstance(row,dict)or set(row)!=keys:raise ValueError('Invalid residual checkpoint record')
            if type(row['refinement'])is not int or row['refinement']!=index:
                raise ValueError('Invalid residual checkpoint ordering')
            if type(row['final'])is not bool or row['final']!=(index==self.expected-1):
                raise ValueError('Invalid final residual checkpoint marker')
            if type(row['converged'])is not bool:raise ValueError('Invalid residual convergence flag')
            rounds=row['coupling']
            if not isinstance(rounds,list):raise ValueError('Invalid interface convergence records')
            if row['final']:
                if not 1<=len(rounds)<=self.config.residual_coupling_maxiter:
                    raise ValueError('Invalid endpoint coupling count')
                for item in rounds:
                    if not isinstance(item,dict)or set(item)!={'force_change_n','torque_change_nm','rigid_translation_update_m','rigid_rotation_update_rad','shell_converged','corrections'}:
                        raise ValueError('Invalid coupling record')
                    for key in ('force_change_n','torque_change_nm','rigid_translation_update_m','rigid_rotation_update_rad'):
                        value=item[key]
                        if type(value)not in (int,float)or not math.isfinite(value)or value<0.:
                            raise ValueError('Invalid interface wrench change')
                    if item['shell_converged']is not True:
                        raise ValueError('Unconverged checkpoint coupling round')
                    if type(item['corrections'])is not int or not 0<=item['corrections']<=self.config.residual_maxiter:
                        raise ValueError('Invalid coupling correction count')
                if rounds[-1]['force_change_n']>INTERFACE_FORCE_TOLERANCE_N or rounds[-1]['torque_change_nm']>INTERFACE_TORQUE_TOLERANCE_NM:
                    raise ValueError('Unreconciled checkpoint interface')
                if rounds[-1]['rigid_translation_update_m']>RIGID_TRANSLATION_TOLERANCE_M or rounds[-1]['rigid_rotation_update_rad']>RIGID_ROTATION_TOLERANCE_RAD:
                    raise ValueError('Unconverged checkpoint rigid iteration')
                if row['corrections']!=sum(item['corrections']for item in rounds):
                    raise ValueError('Inconsistent coupling correction count')
            elif rounds:raise ValueError('Unexpected intermediate coupling records')
            budget=self.config.residual_maxiter*len(rounds) if row['final']else 1
            if type(row['corrections'])is not int or not 0<=row['corrections']<=budget:
                raise ValueError('Invalid residual correction count')
            for key in ('final_maximum_vertex_N','final_l2_N'):
                value=row[key]
                if type(value)not in (int,float)or not math.isfinite(value)or value<0.:
                    raise ValueError('Invalid checkpoint force residual')
            if row['converged']!=(row['final_maximum_vertex_N']<=FORCE_TOLERANCE_N):
                raise ValueError('Inconsistent checkpoint convergence flag')
            if row['final']and not row['converged']:raise ValueError('Unconverged checkpoint endpoint')

    def restore_checkpoint(self,records):
        self.validate_checkpoint(records)
        self.close()
        self.records=copy.deepcopy(records);self.g.residual_records=self.records
        self.g.refinement_number=len(self.records)

    def close(self):
        if self.solver is not None:self.solver.close()


def install(coupled,config):
    runtime=ResidualStepper(coupled,config)
    coupled.residual_solver=runtime
    coupled.step=runtime.step
    coupled.close=runtime.close
    return coupled

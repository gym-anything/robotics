"""Shared runtime residual-Newton solver for the unchanged shell force equations.

No fictitious energy for nonconservative friction. A colored AD or finite-
difference Jacobian includes normal/friction cross derivatives. Residual-norm
search uses its actual directional derivative; pseudo-transient search checks
the defect of its artificial implicit equation instead. Collision bounds and
the final unmodified physical-force criterion remain mandatory.

Used by the explicit residual-shell-v5 environment revision and offline probes.
Source integration is NOT an executed physical or performance qualification.

Basis: PETSc SNES Newton line search / SNESComputeJacobianDefaultColor.
"""
import time
import numpy as np
import warp as wp
from scipy.sparse import coo_matrix,eye
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_spring_force_and_hessian_both_vertices


@wp.kernel
def precise_spring_correction(dt:float,q:wp.array[wp.vec3],previous:wp.array[wp.vec3],
    exact:wp.array[wp.vec3d],old_exact:wp.array[wp.vec3d],indices:wp.array[int],
    rest:wp.array[float],stiffness:wp.array[float],damping:wp.array[float],force:wp.array[wp.vec3]):
    i=wp.tid();a,b,fa,fb,H=evaluate_spring_force_and_hessian_both_vertices(i,dt,q,previous,indices,rest,stiffness,damping)
    dx=exact[a]-exact[b];length=wp.max(wp.length(dx),wp.float64(1e-8))
    old_length=wp.max(wp.length(old_exact[a]-old_exact[b]),wp.float64(1e-8))
    value=wp.float64(stiffness[i])*(wp.float64(rest[i])/length-wp.float64(1.))*dx
    value-=dx/length*wp.float64(damping[i]/dt)*(length-old_length)
    f=wp.vec3(float(value[0]),float(value[1]),float(value[2]))
    wp.atomic_add(force,a,f-fa);wp.atomic_add(force,b,-f-fb)


class ResidualNewton:
    def __init__(self,g,state,contacts,dt,*,audit_derivatives=True):
        self.g=g;self.state=state;self.contacts=contacts;self.dt=dt
        self.m=g.s.model;self.free=np.flatnonzero(self.m.particle_mass.numpy()>0)
        self.dofs=(3*self.free[:,None]+np.arange(3)).ravel()
        self.graph=self.pattern();self.colors=group_columns(self.graph,order=0)
        self.groups=[np.flatnonzero(self.colors==i)for i in range(self.colors.max()+1)]
        self.calls=0;self.audit_derivatives=audit_derivatives
        self.cache=None

    def pattern(self):
        m=self.m;s=self.g.s;n=m.particle_count
        blocks=[]
        def add(a):
            a=np.asarray(a,dtype=np.int64)
            if not a.size:return
            if a.ndim==1:a=a[:,None]
            keys=(a[:,:,None]*n+a[:,None,:]).reshape(-1)
            valid=((a[:,:,None]>=0)&(a[:,None,:]>=0)).reshape(-1)
            blocks.append(keys[valid])
        add(np.arange(n));add(m.tri_indices.numpy())
        edges=m.edge_indices.numpy();add(edges[(edges>=0).all(1)])
        if m.spring_count:add(m.spring_indices.numpy().reshape(-1,2))
        c=self.contacts;count=int(c.soft_contact_count.numpy()[0])
        if not 0<=count<=c.soft_contact_max:
            raise OverflowError('Cannot color truncated rigid/surface contacts')
        add(c.soft_contact_indices.numpy()[:count])
        d=s.trimesh_collision_detector;tri=m.tri_indices.numpy()
        for kind in ('vertex_colliding_triangles','edge_colliding_edges'):
            counts=getattr(d,kind+'_count').numpy();offsets=getattr(d,kind+'_offsets').numpy()
            caps=getattr(d,kind+'_buffer_sizes').numpy();pairs=getattr(d,kind).numpy().reshape(-1,2)
            if np.any(counts>caps)or np.any(counts<0):raise OverflowError('Cannot color truncated contact candidates')
            directed=set()
            rows=[]
            for i in np.flatnonzero(counts):
                others=pairs[offsets[i]:offsets[i]+counts[i],1];others=others[others>=0]
                if not len(others):continue
                if len(np.unique(others))!=len(others):raise ValueError('Duplicate self-contact candidate')
                if kind.startswith('vertex'):
                    rows.append(np.column_stack((np.full(len(others),i),tri[others])))
                else:
                    for other in others:directed.add((int(i),int(other)))
                    rows.append(np.column_stack((np.tile(edges[i,2:],(len(others),1)),edges[others,2:])))
            if kind.startswith('edge') and any((j,i)not in directed for i,j in directed):
                raise ValueError('Nonreciprocal edge candidates would omit a contact reaction')
            if rows:add(np.concatenate(rows))
        keys=np.unique(np.concatenate(blocks));ri=keys//n;ci=keys%n
        rr=(3*ri[:,None,None]+np.arange(3)[None,:,None]+np.zeros((1,1,3),int)).ravel()
        cc=(3*ci[:,None,None]+np.arange(3)[None,None,:]+np.zeros((1,3,1),int)).ravel()
        graph=coo_matrix((np.ones(len(rr),np.int8),(rr,cc)),shape=(3*n,3*n)).tocsr()
        return graph[self.dofs,:][:,self.dofs].tocsc()

    def rebuild_pattern(self):
        self.graph=self.pattern();self.colors=group_columns(self.graph,order=0)
        self.groups=[np.flatnonzero(self.colors==i)for i in range(self.colors.max()+1)]
        self.pattern_dirty=False

    def evaluate(self,x):
        g=self.g;p=g.precision
        p.trial.assign(x);g.trial.assign(x.astype(np.float32))
        if hasattr(g,'accurate_residual'):
            value=g.accurate_residual.evaluate(p.trial,self.state,self.contacts,self.dt)
            self.calls+=1
            return value.numpy().ravel()[self.dofs]
        g.evaluate(g.trial,self.state,self.contacts,self.dt,hessians=False,precise_q=p.trial)
        if self.m.spring_count and not getattr(g,'retention_precision',False):
            wp.launch(precise_spring_correction,self.m.spring_count,[self.dt,g.trial,g.s.particle_q_prev,
                p.trial,p.previous,self.m.spring_indices,self.m.spring_rest_length,self.m.spring_stiffness,
                self.m.spring_damping,g.force],device=g.device)
        self.calls+=1
        return g.force.numpy().astype(float).ravel()[self.dofs]

    def jacobian(self,x,h=2e-9):
        # Absolute SI perturbation, below contact smoothing and geometry scales.
        # Its adequacy is explicitly checked with an independent J*v difference.
        if getattr(self,'dynamic_pattern',False) and getattr(self,'pattern_dirty',True):
            self.rebuild_pattern()
        if getattr(self.g,'analytic_derivatives',False):return self.analytic_jacobian(x)
        indices=self.graph.indices;ptr=self.graph.indptr
        out=np.zeros(len(indices));flat=x.ravel()
        for cols in self.groups:
            up=flat.copy();down=flat.copy();dofs=self.dofs[cols]
            up[dofs]+=h;down[dofs]-=h
            difference=(self.evaluate(up.reshape(x.shape))-self.evaluate(down.reshape(x.shape)))/(2*h)
            for col in cols:out[ptr[col]:ptr[col+1]]=difference[indices[ptr[col]:ptr[col+1]]]
        from scipy.sparse import csc_matrix
        return csc_matrix((out,indices.copy(),ptr.copy()),shape=self.graph.shape)

    def analytic_jacobian(self,x):
        """Colored reverse-mode AD; no finite-difference roundoff in soft modes."""
        g=self.g;p=g.precision;field=g.accurate_residual
        if not field.full_contact:raise ValueError('Full differentiable force assembly required')
        # Independent tape input avoids changing archived or native state arrays.
        q=wp.array(x,dtype=wp.vec3d,device=g.device,requires_grad=True)
        # A new Tape has no registered adjoints until its first backward().
        # Tape.zero() cannot clear persistent force buffers at that point.
        # Otherwise the first color inherits the last previous query's seed.
        for array in (field.force,field.result):
            array.grad.zero_()
        tape=wp.Tape()
        with tape:value=field.evaluate(q,self.state,self.contacts,self.dt)
        indices=self.graph.indices;ptr=self.graph.indptr;out=np.zeros(len(indices))
        seed=wp.zeros_like(value)
        for rows in self.groups:
            tape.zero();incoming=np.zeros_like(x);incoming.ravel()[self.dofs[rows]]=1.
            seed.assign(incoming);tape.backward(grads={value:seed})
            derivative=q.grad.numpy().ravel()[self.dofs]
            for row in rows:out[ptr[row]:ptr[row+1]]=derivative[indices[ptr[row]:ptr[row+1]]]
        from scipy.sparse import csc_matrix
        return csc_matrix((out,indices.copy(),ptr.copy()),shape=self.graph.shape).T.tocsc()

    def close(self):
        if self.cache is not None and self.cache.get('direct') is not None:
            self.cache['direct'].free_memory(everything=True)
        self.cache=None

    def begin_problem(self,state,contacts,dt,*,search_shift=None):
        """Invalidate continuation/factors when the coupled force field changes.

        Reusing them within ONE fixed force problem is allowed. A rigid sweep,
        physical timestep, inertia/history update or changed contact buffer is
        a new problem, even when its Python state object has the same identity.
        A caller may supply only a positive scalar search-step hint; all forces,
        Jacobians, factors and contact graphs are still rebuilt. Every proposed
        correction must pass the new problem's collision and defect checks.
        """
        if not np.isfinite(dt) or dt<=0.:raise ValueError('Positive finite timestep required')
        if search_shift is not None and (not np.isfinite(search_shift) or search_shift<=0.):
            raise ValueError('Search hint must be a positive finite scalar')
        self.close()
        self.state=state;self.contacts=contacts;self.dt=dt
        self.regularization=(1. if search_shift is None else float(search_shift)) if getattr(self.g,'pseudo_transient_search',False) else 0.
        self.lm_damping=1e-3
        self.trust_radius=self.g.s.particle_self_contact_margin*.1
        self.dynamic_pattern=True
        # Validate completeness/reciprocity even if the initial residual already
        # meets tolerance and no Jacobian happens to be needed.
        self.rebuild_pattern()

    def solve(self,*,max_iterations=12,absolute_tolerance=1e-5,initial_x=None,pardiso=False,reuse=False,persistent=False,verbose=True):
        try:
            return self._solve(max_iterations=max_iterations,absolute_tolerance=absolute_tolerance,
                initial_x=initial_x,pardiso=pardiso,reuse=reuse,persistent=persistent,verbose=verbose)
        except BaseException:
            self.close()
            raise

    def _solve(self,*,max_iterations,absolute_tolerance,initial_x,pardiso,reuse,persistent,verbose):
        if not persistent:self.close()
        if type(max_iterations)is not int or max_iterations<1:raise ValueError('Positive exact iteration budget required')
        if not np.isfinite(absolute_tolerance) or absolute_tolerance<=0.:raise ValueError('Positive finite force tolerance required')
        g=self.g;m=self.m;x=g.precision.current.numpy().copy() if initial_x is None else np.asarray(initial_x,float).copy()
        if x.shape!=g.precision.current.shape+(3,) or not np.isfinite(x).all():
            raise ValueError('Invalid reference iterate')
        f=self.evaluate(x)
        if not np.isfinite(f).all():raise FloatingPointError('Nonfinite initial physical residual')
        rows=[];start=time.perf_counter();converged=False;factor=None;refresh=True
        direct=None;J=None;matrix=None
        regularized=getattr(g,'regularized_search',False)
        trust=getattr(g,'trust_region_search',False)
        least_squares=getattr(g,'least_squares_search',False)
        pseudo=getattr(g,'pseudo_transient_search',False)
        if sum(map(bool,(regularized,trust,least_squares,pseudo)))>1:
            raise ValueError('Numerical search strategies are mutually exclusive')
        lm=getattr(self,'lm_damping',1e-3)
        radius=getattr(self,'trust_radius',g.s.particle_self_contact_margin*.1)
        shift=getattr(self,'regularization',1. if pseudo else 0.) if regularized or pseudo else 0.
        metric=np.repeat(m.particle_mass.numpy()[self.free].astype(float)/(self.dt*self.dt),3)
        if persistent and self.cache is not None:
            J=self.cache['J'];matrix=self.cache['matrix'];direct=self.cache['direct'];factor=self.cache['factor']
            refresh=J is None or self.cache.get('refresh',False)
        elif pardiso:
            from pypardiso import PyPardisoSolver
            direct=PyPardisoSolver(mtype=11)
            # Own the handle before factorization can raise.
            self.cache=dict(J=None,matrix=None,direct=direct,factor=None,refresh=True)
        for iteration in range(max_iterations):
            # Native outer sweeps refresh their broad phase, but several inner
            # Newton corrections can exhaust that discovery ball on their own.
            # Re-query at an already accepted position; retain every collision
            # bound, inertia target, previous position and physical clock.
            reference=g.s.pos_prev_collision_detection.numpy().astype(float)
            discovery_distance=float(np.linalg.norm(x-reference,axis=1).max())
            threshold=g.s.particle_self_contact_margin*g.s.particle_conservative_bound_relaxation*.25
            search_refreshed=discovery_distance>=threshold
            if search_refreshed:
                self.state.particle_q.assign(x.astype(np.float32))
                g.s._collision_detection_penetration_free(self.state)
                self.rebuild_pattern();refresh=True
                f=self.evaluate(x)
            if not np.isfinite(f).all():raise FloatingPointError('Nonfinite current physical residual')
            norm=float(np.linalg.norm(f));maximum=float(np.linalg.norm(f.reshape(-1,3),axis=1).max())
            if not np.isfinite(norm):raise FloatingPointError('Physical residual norm overflow')
            if maximum<=absolute_tolerance:converged=True;break
            tick=time.perf_counter();jacobian_seconds=0.;derivative_error=None
            if refresh:
                J=self.jacobian(x);jacobian_seconds=time.perf_counter()-tick
                if not np.isfinite(J.data).all():
                    raise FloatingPointError('Nonfinite force Jacobian; no correction accepted')
                if self.audit_derivatives:
                    rng=np.random.default_rng(209);v=rng.normal(size=len(f));v/=np.max(abs(v))
                    perturb=np.zeros_like(x).ravel();perturb[self.dofs]=v*1e-9
                    fd=(self.evaluate(x+perturb.reshape(x.shape))-self.evaluate(x-perturb.reshape(x.shape)))/(2e-9)
                    jv=J@v;derivative_error=float(np.linalg.norm(jv-fd)/max(np.linalg.norm(fd),1e-30))
            factor_start=time.perf_counter()
            if refresh:
                from scipy.sparse import diags
                if least_squares:
                    normal=J.T@J
                    diagonal=np.maximum(normal.diagonal(),metric**2)
                    matrix=(normal+diags(lm*diagonal)).tocsr()
                else:matrix=(J-diags(shift*metric)).tocsr()
                matrix.sort_indices()
                if pardiso:
                    direct.factorize(matrix)
                else:factor=splu(matrix.tocsc())
            rhs=-(J.T@f) if least_squares else -f
            delta=direct.solve(matrix,rhs) if pardiso else factor.solve(rhs)
            factor_seconds=time.perf_counter()-factor_start
            linear_error=float(np.linalg.norm(matrix@delta-rhs)/max(float(np.linalg.norm(rhs)),1e-30))
            if not np.isfinite(delta).all() or not np.isfinite(linear_error) or linear_error>1e-6:
                if direct is not None:direct.free_memory(everything=True)
                self.cache=None
                raise FloatingPointError('Invalid linear correction; no iterate accepted: relative residual '+str(linear_error))
            newton_max=float(np.max(abs(delta)))
            if trust and newton_max>radius:
                gradient=J.T@f;product=J@gradient
                cauchy=-float(gradient@gradient)/max(float(product@product),1e-300)*gradient
                cauchy_max=float(np.max(abs(cauchy)))
                if cauchy_max>=radius:
                    delta=cauchy*(radius/cauchy_max)
                else:
                    toward=delta-cauchy;moving=abs(toward)>0.
                    limits=(np.sign(toward[moving])*radius-cauchy[moving])/toward[moving]
                    delta=cauchy+min(1.,float(limits.min()))*toward
            direction=np.zeros_like(x);direction.ravel()[self.dofs]=delta
            distance=max(float(np.max(abs(delta))),1e-30)
            direction_error=None
            if self.audit_derivatives:
                fraction=min(1e-3,1e-8/distance)
                directional_fd=(self.evaluate(x+fraction*direction)-self.evaluate(x-fraction*direction))/(2*fraction)
                direction_error=float(np.linalg.norm(J@delta-directional_fd)/max(np.linalg.norm(J@delta),1e-30))
            g.q.assign(x.astype(np.float32));g.delta.assign(direction.astype(np.float32))
            g.s.relative_step.increment(g.q,g.delta,1.,g.displacement,g.trial)
            bound=float(g.s.relative_step.scale.numpy()[0])
            pair_bound=float(g.s.relative_step.diagnostic.numpy()[:,0].min())
            if not np.isfinite(bound) or not 0.<=bound<=1.:
                raise FloatingPointError('Invalid collision step bound')
            accepted=False;alpha=bound;rho=None;pseudo_defect=None
            model_change=J@delta
            slope=float(f@model_change)
            # phi=0.5*R.T*R has directional derivative R.T*J*delta.
            # Substituting -||R||^2 is valid only for an exact Newton direction,
            # not a damped, dogleg or least-squares direction.
            # Scaling a solution of (J-sigma*D)d=-R is NOT a solution
            # for a shorter artificial timestep. In the linear case the
            # old defect was exactly (1-alpha)*R, so alpha<0.75 could never
            # pass. Shorten pseudo-time by raising sigma and RE-SOLVING;
            # do not apply the collision-scaled vector to that equation.
            collision_limited=bool(pseudo and bound<1.)
            if collision_limited:alpha=0.
            trials=(0 if collision_limited else 1) if pseudo else 24
            for _ in range(trials):
                trial=x+alpha*direction;value=self.evaluate(trial)
                if alpha<=0. or np.array_equal(trial,x):break
                predicted=norm**2-float(np.linalg.norm(f+alpha*model_change))**2
                actual=norm**2-float(np.linalg.norm(value))**2
                rho=actual/predicted if predicted>0. else -1.
                if pseudo:
                    # Artificial backward-Euler equation for this trial is
                    # G(delta)=R(x+delta)-shift*D*delta=0. The shifted linear
                    # solve is only one Newton approximation to that equation.
                    # Check its ACTUAL defect; finite values alone do not make
                    # that approximation acceptable. This permits physical
                    # residual growth without accepting an inaccurate pseudo
                    # step. 0.25 is a numerical forcing term, not a force gate.
                    defect=value-shift*metric*(alpha*delta)
                    pseudo_defect=float(np.linalg.norm(defect)/max(norm,1e-30))
                    merit=np.isfinite(pseudo_defect) and pseudo_defect<=.25
                else:
                    merit=slope<0. and actual>=-2e-4*alpha*slope and (not trust or rho>=.001)
                if np.isfinite(value).all() and merit:
                    x=trial;f=value;accepted=True;break
                if not pseudo:alpha*=.5
            row=dict(iteration=iteration,before_l2_N=norm,after_l2_N=float(np.linalg.norm(f)),
                maximum_vertex_residual_N=float(np.linalg.norm(f.reshape(-1,3),axis=1).max()),
                derivative_relative_error=derivative_error,linear_relative_residual=linear_error,
                newton_direction_relative_error=direction_error,
                alpha=alpha,collision_bound=bound,accepted=accepted,seconds=time.perf_counter()-tick,
                pair_collision_bound=pair_bound,discovery_distance_m=discovery_distance,
                contact_search_refreshed=search_refreshed,
                search_regularization=shift,direction_max_m=distance,
                least_squares_damping=lm if least_squares else None,
                merit_directional_derivative=slope,pseudo_equation_relative_defect=pseudo_defect,
                rejection_reason=None if accepted else 'pseudo_collision_resolve' if collision_limited else 'pseudo_equation_defect' if pseudo else 'no_descent_step',
                trust_radius_m=radius if trust else None,trust_rho=rho,raw_newton_max_m=newton_max,
                jacobian_seconds=jacobian_seconds,factor_seconds=factor_seconds,jacobian_refreshed=refresh)
            rows.append(row)
            import json
            if verbose:print(json.dumps(row),flush=True)
            next_shift=shift
            if pseudo:
                # Switched-evolution relaxation on the artificial time only.
                # A collision rejection reduces pseudo-time through a new solve.
                # A rejected nearly unshifted Newton step needs a real restart:
                # multiplying an exhausted 1e-12 shift leaves the same cycle.
                if collision_limited:next_shift=max(1.,shift*max(2.,1./max(bound,1e-6)))
                elif not accepted:next_shift=max(1.,shift*10.)
                else:next_shift=shift*float(np.linalg.norm(f))/max(norm,1e-30)/(1.1*alpha)
                shift=max(next_shift,1e-12)
                if shift>1e12:break
                refresh=True
                continue
            elif least_squares:
                if not accepted or alpha<.25 or rho<.25:lm*=10.
                elif rho>.75:lm*=.25
                if lm>1e12:break
                refresh=True
                continue
            elif regularized:
                if not accepted or alpha<.1 or np.linalg.norm(f)>.99*norm:
                    next_shift=max(1.,shift*10.)
                elif alpha>.9 and np.linalg.norm(f)<.7*norm:
                    next_shift=shift*.25
                    if next_shift<1e-8:next_shift=0.
                if next_shift>1e12:break
            elif trust:
                if not accepted or alpha<.25 or rho<.25:radius*=.25
                elif rho>.75 and distance>.9*radius:radius=min(2.*radius,g.s.particle_self_contact_margin*.5)
                if radius<1e-14:break
            elif not accepted and refresh:break
            refresh=bool(not reuse or not accepted or np.linalg.norm(f)>.75*norm or next_shift!=shift)
            shift=next_shift
        self.regularization=shift
        self.lm_damping=lm
        self.trust_radius=radius
        # Local query endpoint only; no simulated time, velocities or body writes.
        f=self.evaluate(x)
        if not np.isfinite(f).all():raise FloatingPointError('Nonfinite final physical residual')
        converged=bool(np.linalg.norm(f.reshape(-1,3),axis=1).max()<=absolute_tolerance)
        if persistent:self.cache=dict(J=J,matrix=matrix,direct=direct,factor=factor,refresh=refresh)
        else:self.close()
        return dict(converged=converged,iterations=rows,calls=self.calls,colors=len(self.groups),
            jacobian_nonzeros=self.graph.nnz,seconds=time.perf_counter()-start,
            final_l2_N=float(np.linalg.norm(f)),final_maximum_vertex_N=float(np.linalg.norm(f.reshape(-1,3),axis=1).max()),
            maximum_iterate_m=float(np.linalg.norm(x-g.precision.current.numpy(),axis=1).max())),x

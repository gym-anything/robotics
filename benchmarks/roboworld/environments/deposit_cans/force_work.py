"""Diagnostic line-work audit of the unchanged implicit force field.

An approximate Newton direction need not decrease a force-norm merit.
For conservative terms -integral(F.dx) is their exact potential difference;
contact friction/damping are retained in the evaluated field, not removed.
Three/five-point Gauss estimates expose quadrature disagreement. This is a
candidate acceptance calculation, not an energy conservation guarantee for
friction or proof of convergence. Physical rollout gates remain mandatory.
"""
import warp as wp
from .relative_step import double

@wp.kernel
def interpolate(q:wp.array[wp.vec3],end:wp.array[wp.vec3],t:float,out:wp.array[wp.vec3]):
    i=wp.tid();out[i]=q[i]+t*(end[i]-q[i])

@wp.kernel
def work(q:wp.array[wp.vec3],end:wp.array[wp.vec3],force:wp.array[wp.vec3],
         w3:wp.float64,w5:wp.float64,out:wp.array[wp.float64]):
    i=wp.tid();v=wp.dot(double(force[i]),double(end[i])-double(q[i]))
    wp.atomic_add(out,0,w3*v);wp.atomic_add(out,1,w5*v)

@wp.kernel
def choose(values:wp.array[wp.float64],accepted:wp.array[int],scale:float,chosen:wp.array[float]):
    error=wp.abs(values[1]-values[0])
    # Require the estimated decrease to exceed TEN times the discrepancy,
    # rather than asking a noisy float32 force field for0.1% quadrature.
    # This is a sign/reliability check for the candidate, not a relaxed
    # physical acceptance gate. Actual energy/force/motion are tested apart.
    accepted[0]=int(chosen[0]==0. and values[1]>wp.float64(10.)*error+wp.float64(1e-10))
    if accepted[0]!=0:chosen[0]=scale

@wp.kernel
def secondary_needed(values:wp.array[wp.float64],need:wp.array[int],counts:wp.array[int]):
    take=values[1]>wp.float64(1e-10) or not wp.isfinite(values[1])
    need[0]=int(take)
    wp.atomic_add(counts,0,1)
    if not take:wp.atomic_add(counts,1,1)

@wp.kernel
def add_midpoint(values:wp.array[wp.float64],midpoint:wp.array[wp.float64]):
    values[0]+=midpoint[0]

class ForceWork:
    def __init__(self,g):
        self.g=g;self.q=wp.zeros_like(g.q)
        self.values=wp.zeros(2,dtype=wp.float64,device=g.device)
        self.midpoint=wp.zeros_like(self.values)
        self.need_secondary=wp.zeros(1,dtype=int,device=g.device)
        self.shortcut_counts=wp.zeros(2,dtype=int,device=g.device)
    def evaluate(self,end,state,contacts,dt):
        g=self.g;m=g.s.model;self.values.zero_()
        if getattr(g,'work_shortcuts',False):
            self._shortcuts(end,state,contacts,dt);return
        # Coincident midpoint is evaluated once for both orders.
        nodes=((.046910077030668,0.,.118463442528095),
               (.112701665379258,5/18,0.),
               (.230765344947159,0.,.239314335249683),
               (.5,4/9,.284444444444444),
               (.769234655052841,0.,.239314335249683),
               (.887298334620742,5/18,0.),
               (.953089922969332,0.,.118463442528095))
        for t,w3,w5 in nodes:
            if g.precision:
                from .elastic_precision import interpolate as exact_interpolate,work as exact_work
                p=g.precision
                wp.launch(exact_interpolate,m.particle_count,[p.reference,p.trial,wp.float64(t),p.sample,self.q],device=g.device)
                g.evaluate(self.q,state,contacts,dt,hessians=not g.force_only_evaluation,precise_q=p.sample)
                wp.launch(exact_work,m.particle_count,[p.reference,p.trial,g.force,wp.float64(w3),wp.float64(w5),self.values],device=g.device)
            else:
                wp.launch(interpolate,m.particle_count,[g.q,end,t,self.q],device=g.device)
                g.evaluate(self.q,state,contacts,dt,hessians=not g.force_only_evaluation)
                wp.launch(work,m.particle_count,[g.q,end,g.force,wp.float64(w3),wp.float64(w5),self.values],device=g.device)

    def _shortcuts(self,end,state,contacts,dt):
        # EXACT same Gauss3/5 inequality: W5<=floor can never be accepted,
        # regardless of W3. Shared midpoint force is evaluated only once.
        from .elastic_precision import interpolate as sample_position,work as accumulate
        g=self.g;m=g.s.model;p=g.precision;self.midpoint.zero_()
        def sample(t,w3,w5,destination):
            wp.launch(sample_position,m.particle_count,[p.reference,p.trial,wp.float64(t),p.sample,self.q],device=g.device)
            g.evaluate(self.q,state,contacts,dt,hessians=False,precise_q=p.sample)
            wp.launch(accumulate,m.particle_count,[p.reference,p.trial,g.force,wp.float64(w3),wp.float64(w5),destination],device=g.device)
        for t,w5 in ((.046910077030668,.118463442528095),(.230765344947159,.239314335249683),
                     (.5,.284444444444444),(.769234655052841,.239314335249683),
                     (.953089922969332,.118463442528095)):
            sample(t,0.,w5,self.values)
            if t==.5:
                wp.launch(accumulate,m.particle_count,[p.reference,p.trial,g.force,wp.float64(4/9),wp.float64(0.),self.midpoint],device=g.device)
        wp.launch(secondary_needed,1,[self.values,self.need_secondary,self.shortcut_counts],device=g.device)
        def secondary():
            sample(.112701665379258,5/18,0.,self.values)
            wp.launch(add_midpoint,1,[self.values,self.midpoint],device=g.device)
            sample(.887298334620742,5/18,0.,self.values)
        wp.capture_if(self.need_secondary,secondary)

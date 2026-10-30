"""Explicit SI configuration."""
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class Config:
    device: str = 'cuda:0'
    solver_revision: str = 'partitioned-v1'
    seed: int = 0
    scene_revision: int = 1
    target_count: int = 24
    tub_count: int = 16
    preload_count: int = 48
    dt: float = 1/960
    iterations: int = 80
    rigid_iterations: int = 40
    proxy_relaxation: float = 1.
    coupling_iterations: int = 1
    shell_numerics: str = 'local'
    global_start: str = 'previous'
    global_merit: str = 'norm'
    global_line_trials: int = 6
    global_stride: int = 16
    rigid_compression_reference: bool = False
    linear_maxiter: int = 160
    residual_maxiter: int = 80
    residual_coupling_maxiter: int = 16
    residual_linear_solver: str = 'pardiso'
    force_only_evaluation: bool = False
    work_shortcuts: bool = False
    batched_elastic_force: bool = False
    cache_elastic_search: bool = False
    tiled_search_reductions: bool = False
    elastic_precision: bool = False
    reciprocal_topology: bool = True
    bag_local_contact: bool = False
    can_segments: int = 24
    bag_segments: int = 64
    bag_rows: int = 36
    film_thickness: float = 38.1e-6
    film_density: float = 920.
    film_young: float = 200e6
    film_poisson: float = .45
    aluminum_young: float = 69e9
    aluminum_poisson: float = .33
    aluminum_yield: float = 220e6
    contact_radius: float = .0005
    contact_margin: float = .005
    rigid_soft_margin: float = .005
    contact_ke: float = 10000.
    contact_kd: float = .02
    friction: float = .3
    friction_velocity: float = .0005
    retention_compliance: float = 1e-5
    gripper_force: float = 10.
    gripper_speed: float = .05
    capture: bool = True
    plasticity: bool = True
    contact_capacity: int = 65536
    # Task contract chosen by the catalogue (variants.resolve). '' keeps each inventory's standing contract
    # (task.task_contract), so fixtures, probes and earlier records are judged exactly as before. No physics effect.
    task_contract: str = ''

    def __post_init__(self):
        if type(self.task_contract) is not str:raise TypeError('task_contract must be a string')
        if self.solver_revision not in ('partitioned-v1','implicit-shell-v2','implicit-shell-v3','implicit-shell-v4','residual-shell-v5','bounded-shell-v6'):
            raise ValueError('Unknown solver revision')
        if (self.solver_revision.startswith('implicit-shell-')or self.solver_revision in ('residual-shell-v5','bounded-shell-v6')) and (self.shell_numerics!='global' or self.global_merit!='work' or not self.elastic_precision):
            raise ValueError('Implicit shell solvers require global work merit and precise elastic coordinates')
        if type(self.capture)is not bool:raise TypeError('Capture must be boolean')
        if type(self.batched_elastic_force)is not bool:raise TypeError('Batched elastic force must be boolean')
        if self.batched_elastic_force and self.solver_revision!='bounded-shell-v6':raise ValueError('Batched force assembly is a v6 candidate')
        if type(self.work_shortcuts)is not bool:raise TypeError('Work shortcuts must be boolean')
        if self.work_shortcuts and (self.solver_revision!='bounded-shell-v6' or not self.force_only_evaluation):
            raise ValueError('Same-rule work shortcuts require explicit v6 force-only evaluation')
        if self.solver_revision=='residual-shell-v5' and self.capture:
            raise ValueError('residual-shell-v5 requires capture=False; host sparse solves cannot be CUDA-captured')
        if type(self.residual_maxiter)is not int or not 1<=self.residual_maxiter<=1024:
            raise ValueError('Residual correction budget must be1..1024')
        if type(self.residual_coupling_maxiter)is not int or not 1<=self.residual_coupling_maxiter<=1024:
            raise ValueError('Endpoint coupling budget must be1..1024')
        if self.residual_linear_solver not in ('pardiso','superlu'):
            raise ValueError('Unknown residual linear solver')
        if type(self.global_stride)is not int or not 1<=self.global_stride<=64:raise ValueError('Global stride must be1..64')
        if type(self.rigid_compression_reference)is not bool:raise TypeError('Compression reference must be boolean')
        if self.elastic_precision and (self.shell_numerics!='global' or self.global_start!='previous'):
            raise ValueError('Precise coordinates require global solve with previous-state initial iterate')
        if type(self.global_line_trials) is not int or not 6<=self.global_line_trials<=16:raise ValueError('Line trials must be6..16')
        if type(self.linear_maxiter) is not int or not 1<=self.linear_maxiter<=1024:
            raise ValueError('Linear iteration budget must be1..1024')
        if type(self.scene_revision) is not int or self.scene_revision not in (1,2,3,4,5,6,7):raise ValueError('Unknown scene revision')
        if self.scene_revision>=6 and self.tub_count>16:raise ValueError('Authored pocket layout supports at most16 tub cans')
        if self.global_merit not in ('norm','work'):raise ValueError('Unknown implicit merit')
        if self.global_start not in ('previous','predictor'):raise ValueError('Unknown implicit initial guess')
        if self.shell_numerics not in ('local','global'):raise ValueError('Unknown shell numerical method')
        for k,v in asdict(self).items():
            if isinstance(v,float) and not math.isfinite(v):raise ValueError(f'Nonfinite {k}')
        for k in ('target_count','tub_count','preload_count','seed'):
            if type(getattr(self,k)) is not int or getattr(self,k)<0:raise ValueError(k+' must be nonnegative integer')
        if not 0<=self.tub_count<=self.target_count:raise ValueError('Tub count exceeds target inventory')
        for k in ('iterations','rigid_iterations','can_segments','bag_segments','bag_rows','contact_capacity'):
            if type(getattr(self,k)) is not int or getattr(self,k)<1:raise ValueError('Positive integer '+k+' required')
        if self.can_segments<16 or self.bag_segments%8 or self.bag_segments<32:raise ValueError('Invalid surface mesh resolution')
        if type(self.coupling_iterations) is not int or not 1<=self.coupling_iterations<=16:raise ValueError('Coupling iterations must be1..16')
        if self.dt<=0 or min(self.film_thickness,self.film_density,self.film_young,self.aluminum_young,self.aluminum_yield)<=0:raise ValueError('Positive physical constants required')
        if not 0<self.film_poisson<.5 or not 0<self.aluminum_poisson<.5:raise ValueError('Invalid Poisson ratio')
        if not 5<=self.gripper_force<=80 or not 0<self.gripper_speed<=.420:raise ValueError('WSG operating range exceeded')
        if self.contact_ke<=0 or self.contact_kd<0 or self.friction<0 or self.friction_velocity<=0:
            raise ValueError('Contact stiffness/regularization must be positive; damping/friction nonnegative')
        if self.retention_compliance<=0 or self.contact_radius<=0 or self.contact_margin<=self.contact_radius:raise ValueError('Invalid contact/retention parameters')
        if self.rigid_soft_margin<=self.contact_radius:raise ValueError('Rigid/soft discovery margin must exceed physical contact radius')
        if not 0<self.proxy_relaxation<=1:raise ValueError('Proxy relaxation must be in (0,1]')


def resolved_contact_capacity(config):
    """Retain the reference scene's contact storage per can-shell inventory."""
    reference_cans=72;reference_capacity=65536
    cans=config.target_count+config.preload_count
    required=0 if cans<=reference_cans else math.ceil(reference_capacity*cans/reference_cans)
    return max(config.contact_capacity,required)

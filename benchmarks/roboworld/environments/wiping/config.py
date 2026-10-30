"""SI units. Product specifications and engineering estimates are distinguished
in PARAMETERS.md; these defaults are qualification candidates, not measured fits.
"""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class Config:
    device: str = "cuda:0"
    seed: int = 0
    dt: float = 1 / 600
    nonlinear_iterations: int = 12
    linear_iterations: int = 40
    rigid_iterations: int = 30
    contact_capacity: int = 32768
    table_height: float = .90
    table_width: float = 1.2192
    table_depth: float = .762
    backsplash_height: float = .127
    cloth_side: float = .40
    cloth_edge: float = .010
    cloth_dry_mass: float = .061
    cloth_radius: float = .0015
    # Candidate discovery horizon; separate from physical pile radius/forces.
    # Older recordings used the formerly hard-coded3mm value.
    cloth_contact_margin: float = .003
    cloth_count: int = 3
    # Native VBD 2D Lame coefficients (N/m), kept explicit for provenance.
    # Historical defaults remain readable for old recordings.
    membrane_shear: float = 25.
    membrane_lambda: float = 120.
    membrane_damping: float = .05
    initial_water_fraction: float = .75
    water_capacity_fraction: float = 5.
    contact_ke: float = 1500.
    contact_kd: float = 2.
    contact_mu: float = .45
    friction_velocity: float = .01
    joint_limit_ke: float = 10000.
    joint_limit_kd: float = 10.
    gripper_effort: float = 20.
    gripper_stiffness: float = 3500.
    gripper_speed: float = .05
    capture: bool = True
    cloth_solver: str = "style3d"
    scene: str = "component"
    soil: bool = False
    film_transport: bool = False
    film_hz: int = 60
    wrist_cameras: bool = False
    soil_preset: str = 'reference'
    coupling: str = 'lagged'
    native_rigid_iterations: int = 20
    native_rigid_phase: str = 'first'
    # Repeated coupled solves over the SAME timestep, not additional time.
    # Keep the historical one-pass baseline unless explicitly requested.
    native_coupling_iterations: int = 1
    native_adaptive_initial_guess: bool = False
    native_local_frame_z: float = 0.
    native_translation_refinement: bool = False
    native_rigid_contact_onset_correction: bool = False
    native_grain_solver: str = 'mujoco'
    # Keep all articulated and free rigid bodies in one constrained solver.
    # "split" remains available to deserialize the historical qualification runs.
    native_rigid_ownership: str = 'split'
    rigid_contact_history: bool = False
    scene_revision: int = 0
    self_contact_filter_hops: int = 1
    self_contact_material_exclusion: float = 0.
    native_rigid_linear_beta: float = 0.
    # Native Newton1.5 body-contact C0 stabilization. Historical default was
    # implicit0.95; record it explicitly before testing the cold-contact path.
    native_rigid_contact_alpha: float = .95
    native_contact_seed_mass_scale: float = 0.
    native_proxy_finger_joints: bool = False
    native_sphere_contact_refresh: bool = False
    native_mesh_contact_refresh: bool = False
    native_shape_grouped_contacts: bool = False
    native_analytic_box_contacts: bool = False
    native_stable_box_contacts: bool = False
    native_contact_compression_core: bool = False
    native_self_contact_increments: bool = False
    # Collision candidate horizon only; does not enlarge the physical sphere.
    # Preserve historical 1mm default when opening older recorded configurations.
    grain_contact_gap: float = .001
    native_rigid_contact_damping_ratio: float = 0.
    native_zero_slip_hessian: str = 'off'

    def __post_init__(self):
        if type(self.native_rigid_contact_onset_correction) is not bool:raise ValueError('Contact onset correction must be boolean')
        if type(self.native_translation_refinement) is not bool:raise ValueError('Translation refinement must be boolean')
        if self.native_translation_refinement and not self.native_local_frame_z:
            raise ValueError('Translation refinement requires the precise local frame')
        if self.native_local_frame_z and (self.coupling!='native' or self.native_coupling_iterations!=1):
            raise ValueError('Local numerical frame requires one native coupling pass')
        if type(self.native_adaptive_initial_guess) is not bool:raise ValueError('Adaptive guess must be boolean')
        if self.native_adaptive_initial_guess and (self.coupling!='native' or self.native_coupling_iterations!=1):
            raise ValueError('Adaptive initial-guess diagnostic requires one native coupling pass')
        if not 0<=self.native_rigid_contact_alpha<=1:raise ValueError('Rigid contact alpha must be0..1')
        if self.native_zero_slip_hessian not in ('off','body','both'):raise ValueError('Unknown zero-slip Hessian candidate')
        if self.native_zero_slip_hessian!='off' and self.cloth_solver!='vbd':raise ValueError('Zero-slip candidate requires VBD')
        if self.native_rigid_phase not in ('first','last'):raise ValueError('Unknown native rigid pass phase')
        if type(self.scene_revision) is not int or self.scene_revision not in (0,1,2,3,4):raise ValueError('Unknown scene revision')
        if type(self.rigid_contact_history) is not bool:raise ValueError('rigid_contact_history must be boolean')
        if type(self.native_proxy_finger_joints) is not bool:raise ValueError('native_proxy_finger_joints must be boolean')
        if type(self.native_sphere_contact_refresh) is not bool:raise ValueError('native_sphere_contact_refresh must be boolean')
        if type(self.native_mesh_contact_refresh) is not bool:raise ValueError('native_mesh_contact_refresh must be boolean')
        if type(self.native_shape_grouped_contacts) is not bool:raise ValueError('native_shape_grouped_contacts must be boolean')
        if type(self.native_analytic_box_contacts) is not bool:raise ValueError('native_analytic_box_contacts must be boolean')
        if type(self.native_stable_box_contacts) is not bool:raise ValueError('native_stable_box_contacts must be boolean')
        if self.native_stable_box_contacts and not self.native_analytic_box_contacts:raise ValueError('Stable box manifolds require analytic box routing')
        if type(self.native_contact_compression_core) is not bool:raise ValueError('native_contact_compression_core must be boolean')
        if type(self.native_self_contact_increments) is not bool:raise ValueError('native_self_contact_increments must be boolean')
        if self.native_self_contact_increments and self.cloth_solver!='vbd':raise ValueError('Self-contact increment arithmetic requires VBD')
        if self.native_contact_compression_core and (self.cloth_solver!='vbd' or self.coupling!='native' or abs(self.cloth_radius-.0015)>1e-10):
            raise ValueError('Compression candidate requires native VBD with the1.5mm cloth pile radius')
        if self.native_mesh_contact_refresh and (self.cloth_solver!='vbd' or self.coupling!='native'):
            raise ValueError('Mesh refresh requires the native VBD rigid/cloth solve')
        if type(self.self_contact_filter_hops) is not int or not 1<=self.self_contact_filter_hops<=3:raise ValueError('Self-contact filter must be1..3 rings')
        if not 0<=self.self_contact_material_exclusion<=.01:raise ValueError('Invalid material-space self-contact exclusion')
        if not 0<=self.native_rigid_linear_beta<=1e7:raise ValueError('Invalid rigid contact penalty ramp')
        if not 0<=self.native_contact_seed_mass_scale<=1e8:raise ValueError('Invalid mass-scaled rigid contact seed')
        if self.native_contact_seed_mass_scale and not self.native_rigid_linear_beta:raise ValueError('Mass-scaled seed requires native penalty ramp')
        for name in ('nonlinear_iterations','linear_iterations','rigid_iterations','native_rigid_iterations','native_coupling_iterations','contact_capacity','film_hz'):
            value=getattr(self,name)
            if type(value) is not int or value<=0:raise ValueError(f'{name} must be a positive integer')
        for key, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Non-finite {key}")
        if self.dt <= 0 or self.cloth_edge <= 0 or self.cloth_count not in (1, 3):
            raise ValueError("Invalid timestep, mesh spacing or cloth inventory")
        if self.friction_velocity<=0:raise ValueError('Positive friction smoothing velocity required')
        if min(self.joint_limit_ke,self.joint_limit_kd)<=0:raise ValueError('Joint-stop stiffness/damping must be positive')
        if not 0<self.grain_contact_gap<=.05:raise ValueError('Invalid grain collision detection gap')
        if not 0<self.cloth_contact_margin<=.05:raise ValueError('Invalid cloth collision detection horizon')
        if not 0<=self.native_rigid_contact_damping_ratio<=1:raise ValueError('Invalid rigid pair damping ratio')
        if self.membrane_shear<=0 or self.membrane_lambda<=0 or self.membrane_damping<0:
            raise ValueError('Positive membrane stiffness and nonnegative damping required')
        if abs(round(self.cloth_side/self.cloth_edge)*self.cloth_edge-self.cloth_side) > 1e-8:
            raise ValueError("Cloth spacing must exactly divide its side")
        if not 0 < self.gripper_effort <= 35 or not 0 < self.gripper_speed <= .05:
            raise ValueError("Franka Hand force/speed limit exceeded")
        if not 0 < self.gripper_stiffness <= 20000:raise ValueError('Invalid gripper servo stiffness')
        if self.cloth_solver not in ('style3d','vbd'):
            raise ValueError('Unknown cloth solver')
        if self.scene not in ('component','kitchen'):
            raise ValueError('Unknown scene')
        if self.soil_preset not in ('reference','oil-heavy','sticky-food'):raise ValueError('Unknown soil preset')
        ticks=1/(self.film_hz*self.dt)
        if ticks<1 or abs(ticks-round(ticks))>1e-8:raise ValueError('Film period must be an integer physics-tick count')
        if self.film_transport and not self.soil:raise ValueError('Film flow requires material reservoirs')
        if self.coupling not in ('lagged','native'):raise ValueError('Unknown coupling')
        if self.native_grain_solver not in ('mujoco','vbd'):raise ValueError('Unknown native grain solver')
        if self.native_rigid_ownership not in ('split','mujoco'):raise ValueError('Unknown native rigid ownership')
        if self.native_rigid_ownership=='mujoco' and self.native_grain_solver!='mujoco':
            raise ValueError('Unified MuJoCo rigid ownership includes grains')
        if self.coupling=='native' and (self.cloth_solver!='vbd' or self.nonlinear_iterations%self.native_rigid_iterations):
            raise ValueError('Native coupling needs VBD and an exact cloth/rigid iteration ratio')

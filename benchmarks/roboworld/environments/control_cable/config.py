"""Explicit SI configuration; product data and estimated mechanics remain distinct."""
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class CableSpec:
    name: str = "sensor"
    length_m: float = 1.5
    diameter_m: float = 0.00470
    line_mass_kg_m: float = 0.040
    axial_rigidity_n: float = 100000.0
    bending_rigidity_nm2: float = 0.00070
    torsional_rigidity_nm2: float = 0.00030
    segments: int = 100
    friction: float = 0.45

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "name":
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid cable {name}: {value}")
        if type(self.segments) is not int or self.segments < 2:
            raise ValueError("Cable segments must be an integer >= 2")

    @property
    def segment_m(self):
        return self.length_m / self.segments

@dataclass(frozen=True)
class Config:
    device: str = "cuda:0"
    dt: float = 1.0 / 960.0
    seed: int = 0
    sensor_port: str = "X1"
    sensor_port_offset_m: float = 0.
    network_length_m: float = 1.
    network_coil_rotation_rad: float = 0.
    sensor_connector_style: str = "straight"
    sensor_socket_roll_rad: float = 0.
    cabinet_width_m: float = .70
    room_style: str = "neutral"
    table_height_m: float = 0.75
    mujoco_iterations: int = 20
    rigid_force_space: bool = False
    rigid_contact_cone: str = "pyramidal"
    driver_contact_damping_ns_m: float = 100.
    vbd_iterations: int = 12
    joint_angular_penalty: float = 100000.0
    rigid_relaxation: float = 1.
    convex_hardware: bool = False
    terminal_anchor_y: float = .0025
    terminal_inward_blade: bool = False
    ferrule_entry_lead_m: float = 0.
    ferrule_square_crimp: bool = False
    terminal_blade_lead_m: float = 0.
    friction_epsilon_m_s: float = .01
    friction_static_tangent_fix: bool = False
    contact_pose_warmstart: bool = False
    m12_negative_entry_chamfer: bool = False
    contact_hard: bool = True
    m12_contact_entry_lead_m: float = 0.
    coupling_iterations: int = 5
    coupling_rho: float = 200.
    cable_linear_hard: bool = False
    cable_constraint_penalty_n_m: float = 100000.
    coupling_contact_matching: str = "disabled"
    m12_spring_contacts: bool = False
    contact_capacity: int = 8192
    body_contact_capacity: int = 512
    contact_gap_m: float = 0.0002
    contact_stiffness_n_m: float = 40000.0
    contact_damping_ns_m: float = 5.0
    gripper_effort_per_finger_n: float = 35.0
    task_revision: int = 1
    cable: CableSpec = CableSpec()

    def __post_init__(self):
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Positive finite timestep required")
        for name in ("mujoco_iterations", "vbd_iterations", "coupling_iterations",
                     "contact_capacity", "body_contact_capacity"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.sensor_port not in ("X1","X2"):raise ValueError("Sensor destination must be X1 or X2")
        if self.room_style not in ("neutral","warm","cool"):raise ValueError("Unknown workshop lighting style")
        if self.sensor_connector_style not in ("straight","angled"):
            raise ValueError("Sensor connector style must be straight or angled")
        for name,lo,hi in (("sensor_port_offset_m",-.02,.02),("network_length_m",.8,1.3),
                           ("sensor_socket_roll_rad",-math.pi,math.pi),
                           ("cabinet_width_m",.70,.80),
                           ("network_coil_rotation_rad",-math.pi,math.pi)):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError(f"Invalid {name}: {value}")
        if type(self.cable_linear_hard) is not bool:raise ValueError("cable_linear_hard must be boolean")
        if type(self.rigid_force_space) is not bool:raise ValueError("rigid_force_space must be boolean")
        if self.rigid_contact_cone not in ("pyramidal","elliptic"):
            raise ValueError("Unknown native rigid contact cone")
        if not math.isfinite(self.driver_contact_damping_ns_m) or self.driver_contact_damping_ns_m<=0:
            raise ValueError("Positive finite driver contact damping required")
        if not math.isfinite(self.cable_constraint_penalty_n_m) or self.cable_constraint_penalty_n_m<=0:
            raise ValueError("Positive finite cable constraint penalty required")
        if self.coupling_contact_matching not in ("disabled","latest","sticky"):
            raise ValueError("Unknown ADMM contact matching mode")
        if isinstance(self.coupling_rho,bool) or not isinstance(self.coupling_rho,(float,int)) or not math.isfinite(self.coupling_rho) or self.coupling_rho<=0:
            raise ValueError("Positive finite ADMM penalty required")
        if not 0 < self.gripper_effort_per_finger_n <= 35:
            raise ValueError("This FR3 hand controller permits 0..35 N per finger")
        if type(self.task_revision) is not int or self.task_revision not in (1, 2):
            raise ValueError("task_revision must be 1 or 2")

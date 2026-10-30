"""Launch configuration: preset, task, seed, robot and numerics. Unknown fields fail (unified contract)."""

from dataclasses import dataclass, fields

from . import params as P

# preset -> tasks
TASKS = {
    "therapy": ("melody", "bordun", "roll", "finger-tap"),
    "gait": ("take-sticks", "steady-beat", "accent-ostinato"),
    "assessment": ("single-tone", "dynamics", "echo", "strike-and-stop"),
    "pentatonic": ("pentatonic-setup",),
}


@dataclass
class Config:
    preset: str = "therapy"
    task: str = "melody"
    seed: int = 0
    robot: str = "dual-fr3"
    device: str = "cpu"                  # Newton model arrays (MuJoCo runs on the CPU; the scene has no GPU physics)
    rigid_dt: float = P.RIGID_DT_S
    acoustics: bool = True               # microphones (False: modes and contacts only, diagnostics)
    room_noise: bool = False             # 35 dBA background noise in the microphone signals
    resonators: bool = True              # xylophone box resonators (False only for the E6 ablation)
    solver_iterations: int = 50
    solver_impratio: float = 3.0         # barcode_scanning value
    solver_noslip_iterations: int = 100
    solver_noslip_tolerance: float = 1e-10
    contact_timeconst: float = 0.004     # s, rigid contact of every non-pad geom (barcode_scanning E17)
    contact_solimp: tuple = (0.95, 0.99, 0.001)
    contact_capacity: int = 4096
    settle_s: float = 0.4                # rigid settle before t = 0 (instruments onto felt, tools into saddles)
    record_trace: bool = True
    policy_samples: int = 16             # path-traced samples per pixel of policy RGB frames
    g1_arm_gravity_ff: bool = True       # G1 waist/arm gravity feed-forward (g1.py; as in barcode_scanning)
    patch_scale: float = 1.0             # contact-patch radius factor (refinement study only; contact.patch_radius)
    vibro_friction: bool = True          # tangential friction of the vibro-contacts (False: frictionless contact,
                                         # kept for a friction ablation)

    def __post_init__(self):
        if self.preset not in TASKS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose {sorted(TASKS)}")
        if self.task not in TASKS[self.preset]:
            raise ValueError(f"Task {self.task!r} is not in preset {self.preset!r}: {TASKS[self.preset]}")
        if type(self.seed) is not int or not 0 <= self.seed < 10000:
            raise ValueError("seed must be an integer 0..9999")
        if abs(self.rigid_dt - P.RIGID_DT_S) > 1e-12:
            raise ValueError("rigid_dt is fixed at 1 ms (48 audio samples per tick; the FR3's 1 kHz command rate)")
        self.contact_solimp = tuple(self.contact_solimp)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**d)

"""Explicit, versioned configuration. Unknown fields fail (dataclass constructor)."""

import os
from dataclasses import dataclass, asdict, field

PRESETS = ("counter", "home", "receiving", "dev")
TASKS = {
    "counter": ("open-bag", "bag-purchase"),
    "home": ("lift-lid", "unwrap-gift", "open-present"),
    "receiving": ("open-carton",),
    "dev": ("pinch",),
}


def default_device():
    """Where the robot simulator's model arrays live: the first GPU, or what GIFT_BAGS_DEVICE says. The robot solver
    is MuJoCo on the CPU either way and the paper's contact world never uses the GPU, so `cpu` runs a session without
    the card: each session otherwise holds about 0.7 GB of GPU memory, and on a card filled by other jobs a launch
    fails with a CUDA out-of-memory error. Images still need the GPU."""
    return os.environ.get("GIFT_BAGS_DEVICE", "cuda:0")


@dataclass(frozen=True)
class Config:
    preset: str = "counter"
    task: str = "open-bag"
    seed: int = 0
    robot: str = "dual-fr3"            # catalog alias (robots/catalog.py ENVIRONMENTS["gift-bags"])
    device: str = field(default_factory=default_device)   # "cuda:0", or GIFT_BAGS_DEVICE (default_device)
    rigid_dt: float = 1e-3             # s, robot (MuJoCo) tick: FR3 1 kHz control
    substeps: int = 4                  # robot ticks per contact-world step (4 ms)
    dhat: float = None                 # m, barrier range; None: the stock's compressible caliper (contact_caliper)
    kappa: float = None                # barrier stiffness; None: calibrated to paper ZD compression (contact_caliper)
    contact_timeconst: float = .005    # s, MuJoCo robot-furniture contact time constant: <= 2 mm transient
                                       # link penetration into furniture; 0.02 s let
                                       # a pressing wrist sink 6.5 mm and load paper with the overlap
    newton_max: int = 24               # Newton iterations per contact step (counted when exceeded)
    newton_tol: float = 1e-5           # m, Newton step tolerance: 2.5 mm/s x 4 ms (IPC's velocity-scaled criterion;
                                       # 1e-6 m cost 2x the iterations for residual forces < 2 N)
    link_k: float = 1e5                # N/m, robot link surface compliance toward its simulated pose
    finger_k: float = 3e5              # N/m, fingertip pad compliance (rubber pad ~1e9 Pa/m over ~3e-4 m^2)
    finger_mu: float = .5              # legacy surface value; the contact world uses params.FRICTION ("finger")
    settle_s: float = .3               # reset: items settle under gravity, robots hold still
    record: bool = True

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"preset must be one of {PRESETS}")
        if self.task not in TASKS[self.preset]:
            raise ValueError(f"task {self.task!r} is not a {self.preset} task: {TASKS[self.preset]}")
        if not (1e-4 <= self.rigid_dt <= 2e-3) or self.substeps < 1:
            raise ValueError("rigid_dt in [0.1, 2] ms and substeps >= 1")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @property
    def dt(self):
        return self.rigid_dt*self.substeps

    def to_dict(self):
        return asdict(self)

"""Launch configuration: preset, task, seed, robot and numerics. Unknown fields fail (unified contract)."""

from dataclasses import dataclass, fields

TASKS = {
    "stockroom": ("count-by-scanning", "scan-location", "count-key-quantity", "hold-scanner"),
    "counter": ("checkout-present", "checkout-bulky", "checkout-counter"),
    "library": ("library-checkin",),
}


RIGID_DT_S = 0.001                   # s, MuJoCo step of the FR3/Panda configurations (the FR3's 1 kHz control rate)
G1_RIGID_DT_S = 0.0005               # s (at 1 ms the unsupported G1 fell over within 15 s of hands-off
                                     # rest and the Dex3 grip slipped 3.7 mm on release; at 0.5 ms it stood 30 s, slipped 0.7 mm)


@dataclass
class Config:
    preset: str = "stockroom"
    task: str = "count-by-scanning"
    seed: int = 0
    robot: str = "dual-fr3"
    device: str = "cuda:0"
    rigid_dt: float = None           # s, MuJoCo step (None: RIGID_DT_S, 1 ms; G1_RIGID_DT_S, 0.5 ms, for the G1)
    solver_iterations: int = 50
    solver_ls_iterations: int = 50   # Newton line-search evaluations per iteration, MuJoCo's own default. The 20 inherited
                                     # from label_application cut a line search short at an arm-on-arm impact (it needed
                                     # 21-22): the solver stopped on an unconverged step, and its residual came out as a
                                     # 1.1-2.3 kN force on the fingers' mimic coupling, which slid a hand's 14-29 g fingers
                                     # sideways at 33-73 m/s, up to 104 mm past their stops
    solver_impratio: float = 3.      # MuJoCo friction/normal impedance ratio. 10 was chosen for grip (a 17.5 N pinch of the
                                     # 214 g scanner crept 0.07 mm/s at 10, 0.11 at 3, 0.12 at 1, with 10 no-slip iterations;
                                     # E7). At 10, a cereal box knocked off a shelf landing on its corner at 4.2 m/s left the
                                     # floor with 3.2x its energy at 1 ms and kept 84 % of it at 0.5 ms; at 3 it keeps 60 %
                                     # at 1 ms, the same as at 0.25 ms with either value (converged)
    solver_noslip_iterations: int = 100  # MuJoCo no-slip post-solver, run to convergence: at MuJoCo's default tolerance (1e-6)
    solver_noslip_tolerance: float = 1e-10  # it stopped before removing the slip, and the counter's basket pile crept 14 um/s
                                     # whatever the iteration count (10, 30, 100); at 1e-10 it crept 15, 6.1, 2.0 and 0.11 um/s
                                     # with 10, 30, 50 and 100 iterations (counter seed 2). Physics cost x1.5 at the
                                     # counter. No-slip itself took a 30 s head pinch from 5.9 to 0.9 deg of pivot (E7)
    solver_update_data_interval: int = 0   # 0: MuJoCo keeps its own float64 state between steps; Newton's state is a float32
                                          # mirror, re-synced into MuJoCo only after direct writes (world.sync_solver). At 1
                                          # (Newton's default: MuJoCo's state overwritten from float32 before every step), 16
                                          # resting bottles and cans walked in straight lines at up to 5.6 um/s; at 0, none
                                          # (worst 0.003 um/s)
    g1_arm_gravity_ff: bool = True   # G1 only: gravity feed-forward torque on the 3 waist and 14 arm joints (world.py;
                                     # Unitree's manipulation stack sends it for the arms)
    contact_timeconst: float = .004  # s, MuJoCo contact time constant of every non-pad geom (rigid; E17, world.py)
    contact_solimp: tuple = (.95, .99, .001)  # MuJoCo contact impedance (d0, dmax, width m) of every non-pad geom
    contact_capacity: int = 4096
    settle_s: float = 1.0            # stock settles under gravity before the task starts (at least this long) ...
    settle_max_s: float = 5.0        # ... and on, 0.1 s at a time, until every free body is slower than settle_speed
    settle_speed: float = .002       # m/s (after 1 s a can lying in the loosely filled basket still rolled 35 mm/s)
    record_trace: bool = True
    render_samples: int = 256        # scanner frames: path-traced samples per pixel per frame, split over its motion-blur
                                     # subframes; Monte Carlo noise 0.67-0.75 % on a white card at 61-121 mm, a third of the
                                     # sensor's at gain 1 (at 24 samples: 6-7 %)
    policy_samples: int = 24         # policy RGB frames

    def __post_init__(self):
        if self.rigid_dt is None:
            self.rigid_dt = G1_RIGID_DT_S if self.robot == "g1" else RIGID_DT_S
        if self.preset not in TASKS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose {sorted(TASKS)}")
        if self.task not in TASKS[self.preset]:
            raise ValueError(f"Task {self.task!r} is not in preset {self.preset!r}: {TASKS[self.preset]}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**d)

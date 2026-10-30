"""Declared SI parameters for the tightening workcell."""

from dataclasses import dataclass
import math


TASKS = ("sensor", "aim", "frame", "mixed", "support", "access", "front-m4", "find-loose")
PRETIGHTENED_TASKS = ("sensor", "aim")


@dataclass(frozen=True)
class Config:
    device: str = "cuda:0"
    dt: float = 1 / 4800
    task: str = "mixed"
    seed: int = 0
    solver_iterations: int = 24
    contact_capacity: int = 4096
    contact_gap_m: float = 0.000025
    # Friction per material pair (all are reference values, none measured on
    # these parts):
    # thread (steel screw in steel nut, electro-zinc): Bossard low end;
    thread_mu: float = 0.14
    # upward M4 head on its steel washer (steel on zinc-plated steel);
    bearing_mu: float = 0.14
    # M3 heads on the W4F's glass-fibre plastic (VISTAL) housing ears;
    m3_bearing_mu: float = 0.30
    # forward M4 head directly on the anodised aluminium bracket leg;
    aluminium_bearing_mu: float = 0.20
    # anodised aluminium on anodised aluminium (rail/bracket/post faces);
    clamp_mu: float = 0.45
    # W4F plastic face on the stainless BEF-W4-A, and the steel clamping
    # strip on the plate's back face.
    sensor_face_mu: float = 0.30
    sensor_strip_mu: float = 0.15
    # Axial joint stiffness targets from an EA/L stack-up:
    # M3 plastic ear column in series with the screw, ~5 MN/m effective;
    # M4 steel/aluminium/T-nut path ~15 MN/m (range 10-50). Each
    # screw bears on two points; a contact pair acts as its two shapes'
    # stiffnesses in series, so stack = 2 / (1/k_point + 1/k_land). The M4
    # points bear on dedicated bracket bearing lands of the same stiffness;
    # the M3 points bear on the plastic ear lands.
    bearing_contact_stiffness_n_m: float = 2.5e7
    # Forward M4: its stack also runs through the bracket-to-post face in
    # series, so its points and lands are stiffer to reach the same range.
    front_bearing_contact_stiffness_n_m: float = 3e7
    bearing_damping_n_s_m: float = 400.
    # Upward-M4 washer/land contact damping (numerical conditioning). The
    # loose drooped rail rests on this contact and chattered at ~600 Hz when
    # it was underdamped (timestep-bound; halving dt cut it 8x). Newton's
    # force-space conversion keeps a contact's time constant above 2 dt only
    # while damping <= 1/(dt f), f = (sum of the two bodies' inverse weights)
    # x (1 - dmax); the 15 g bracket dominated f, capping the damping near
    # 2350 N s/m at a damping ratio ~0.34 (2400: one of two 10 s rests
    # chattered 4-5 um). With the bracket armature below (0.2 kg, f ~0.29),
    # 13000 is a damping ratio ~0.7 with the time constant 0.54 ms (> 2 dt):
    # both 10 s rests settled to 0.12 um (0.1 kg with 6000, ratio ~0.42:
    # one of two rests reached 1.19 um in the last second). Measured.
    top_bearing_damping_n_s_m: float = 13000.
    # The loose rail drops onto that washer through a ~10:1 lever (washer
    # 12.5 mm, rail centre of mass ~129 mm from the pivot edge): pitch
    # stiffness there ~2300 N m/rad on a ~3e-3 kg m2 pivot inertia, critical
    # damping ~5.3 N m s/rad. Undamped, the loose rail rattled for seconds.
    # Viscous damping of 1.6 N m s/rad on the rail's pitch and yaw (damping
    # ratio ~0.3) stands for the dissipation in the loose joint (numerical,
    # disclosed); at hand speeds (<0.05 rad/s) it resists < 0.1 N m.
    rail_rotation_damping_nm_s_rad: float = 1.6
    # Flat aluminium frame interfaces (post, bracket, rail).
    frame_contact_stiffness_n_m: float = 4e6
    frame_contact_damping_n_s_m: float = 400.
    # MuJoCo soft friction creeps under steady shear at ~(1-d)/d. Clamp
    # interfaces use a near-hard impedance, and the solver's frictional-to-
    # normal impedance ratio stiffens every elliptic friction row.
    # Numerical conditioning of the 15 g corner bracket, which is squeezed
    # between stiff contacts (post, washer, rail): added inertia on its three
    # clearance-play coordinates only (gravity and statics are unchanged).
    # 0.2 kg (was 0.01): lets the washer contact above be well damped.
    bracket_armature_kg: float = .2
    bracket_play_damping_n_s_m: float = 20.
    # The same damping on the rail's slot and lift play: without it the resting
    # rail chattered ~1 um at 1.3 mm/s against its limit rows. It adds about
    # 0.06 N viscous drag to the slow loose-rail breakaway ramp.
    rail_play_damping_n_s_m: float = 20.
    clamp_contact_dmax: float = .95
    solver_impratio: float = 1.
    m3_bearing_contact_stiffness_n_m: float = 5e6
    m3_bearing_damping_n_s_m: float = 600.
    # Loaded thread/under-head friction and the sensor clamp are MuJoCo
    # friction-loss rows (implicit, force-bounded) whose capacity is set each
    # tick from the solved loads. The screw armature conditions that row:
    # soft-row creep scales with 1/inertia, and a bare M3 has ~1e-9 kg m2.
    screw_armature_kg_m2: float = 1e-5
    # Self-locking thread (lead angle 3.4-3.6 deg < friction angle 9.2 deg):
    # with nothing touching its head, a screw cannot be turned by axial load,
    # so its rotation is held by a zero-width limit of this stiffness. A bit
    # or finger on the head releases it to the torque/friction-row law.
    screw_lock_stiffness_nm_rad: float = 200.
    screw_self_lock: bool = True
    # A slipping screw sticks again below this speed (Coulomb stick-slip).
    screw_stick_speed_rad_s: float = .02
    # Coulomb stick on the clamp coordinates (rail slide/yaw, bracket x/z).
    clamp_stick: bool = True
    # Which stuck clamp coordinates drop their kinetic friction row: "all",
    # "linear" (rail slide, bracket) or "none" (diagnostic).
    stick_rows_off: str = "all"
    # The self-locking hold also stops the screw's axial carrier (the
    # helical equality alone gave way ~0.19 mm under a sustained 21 N washer
    # pull once the turn was held). Stiffness ~ the steel screw's own axial
    # stiffness; damping 1e-4 s x stiffness (force-space limit stability).
    screw_axial_lock_stiffness_n_m: float = 5e7
    # Upward-M4 shank on the bracket slot ends (the rail's slide stop):
    # 10 mm slot, 4 mm shank (drawing), steel/aluminium contact stiffness.
    rail_slot_half_travel_m: float = .003
    # Forward M4 (4 mm) in the bracket's 4.5 mm hole (drawing).
    bracket_hole_clearance_m: float = .00025
    slot_stop_stiffness_n_m: float = 5e7
    # Damping of the slot, hole and yaw stops as a time x their stiffness.
    # 1e-4 s: at 1e-5 s the loose rail bounces off its slot end and leaves the
    # slot under a 20-50 N push (measured).
    slot_stop_damping_s: float = 1e-4
    # Axial carrier of each screw chain (a 1 g numerical body; the head
    # bearing points ride on it). With 1 N s/m axial damping it sprang
    # against the helical coupling on every washer contact and drove a
    # sustained 20 Hz rattle of the loose rail (0.5 mrad, 155 N impacts);
    # 200 N s/m removes it (rest range 0.019 mrad). Through the thread helix
    # this is only ~2.5e-6 N m s/rad on the screw's rotation.
    screw_axial_damping_n_s_m: float = 200.
    screw_carrier_mass_kg: float = .001
    # Axial armature of each screw's slide: the real screw's rotational
    # inertia (1/2 m r^2 with the head radius: M3 1 g, 2.75 mm -> 3.8e-9;
    # M4 2 g, 3.5 mm -> 1.2e-8 kg m2) reflected through its lead,
    # I (2 pi / pitch)^2 (derived).
    screw_axial_armature_m3_kg: float = .6
    screw_axial_armature_m4_kg: float = 1.0
    # The rail bears on the leg through its flat lip boxes (False) or through
    # rounded lip-edge capsules (True). Kept as a diagnostic switch: capsule
    # contacts pumped energy into the loose rail (it bounced for seconds).
    rail_edge_capsules: bool = False
    # The rail's seat on the bracket leg: one-sided stops of the rail joint
    # (lift >= 0 at the leg edge; pitch >= 0), stiffness per metre of lift
    # (pitch stop: the same over the ~22.5 mm seat half-length). Aluminium
    # rail lips on an aluminium leg: an estimate in the stack-up's range.
    # Force-space limit rows need damping ~1e-4 s x stiffness to stay still
    # here (measured: 1e7 with 300 N s/m walked, with 1000 rested).
    rail_seat_stiffness_n_m: float = 5e7
    rail_seat_damping_n_s_m: float = 1e4
    # Armature on the rail's lift coordinate (numerical conditioning, about
    # the rail's own mass): at the pivot edge that coordinate carries only
    # ~17.5 g (inverse weight 57), too light for a stiff, well-damped seat at
    # this time step (5e7 with 500 N s/m bounced the loose rail; 2e7 with
    # 2000 had its time constant clamped). With it: 5e7 / 1e4, time constant
    # above 2 dt, loose rest still (pitch range 2.2 urad), measured.
    rail_lift_armature_kg: float = .15
    # Nose-up stop (the rail's inboard end pressing into the leg face):
    # stiff enough that a 700 N clamp couple (~9.4 N m) tilts it < 0.1 mrad.
    rail_pitch_stop_nm_rad: float = 1e5
    rail_pitch_stop_damping_nm_s_rad: float = 10.
    # Yaw stops: the key tab's 0.2 mm side clearance at 32.5 mm from the
    # pivot (drawing), and the seat's 2e7 N/m at that lever (2e7 x 0.0325^2).
    rail_yaw_clearance_rad: float = .0002/.0325
    rail_yaw_stop_nm_rad: float = 2.1e4
    # Upward M4 axis to the leg's far edge (the rail pivot): lever of the
    # clamp's yaw friction (drawing: 35.5 - 22.0 mm).
    rail_clamp_lever_m: float = .0135
    # Diagnostic switch: the bracket's vertical play coordinate.
    bracket_vertical_play: bool = True
    # Driver clutch: a one-way detent. Locked (a zero-width joint limit with
    # this torsional stiffness) until the transmitted tightening torque
    # reaches the setting; then it rolls over one notch against a reduced
    # torque and relocks. A "click" is one completed notch; a driver has
    # released on a screw after at least one click with its bit in that
    # socket. Notch angle and roll-over torque are priors (no teardown data
    # for the Wera 7440 / HansTool 0176 cams): six notches per turn and a
    # roll-over torque of 30 % of the setting. Lock stiffness is a numerical
    # choice well above any compliance a policy could resolve (1.75 mrad at
    # 0.35 N m); the small armature conditions the light rotor.
    clutch_armature_kg_m2: float = 1e-5
    clutch_lock_stiffness_nm_rad: float = 200.
    clutch_lock_damping_nm_s_rad: float = .05
    # Detent ramp climbed at the setting torque before the cam goes over the
    # top (prior). The release needs setting x climb of work (0.035 J at
    # 0.35 N m), so a relock impact of ~0.5 mJ cannot re-release it.
    clutch_climb_rad: float = .1
    clutch_notch_rad: float = math.pi/3
    clutch_rollover_fraction: float = .30
    # Frame tasks: the secured rail must be level and square to the bracket
    # (pitch and yaw), about 1/12 of the free droop of a loose corner.
    rail_level_tolerance_rad: float = .003
    # Driver handle contact damping (N s/m), per driver because the 180 g
    # Wera and 38 g HansTool handles see the same contact stiffness with a
    # 4.7x different mass: one fixed 300 gave restitution 0.85 vs 0.32.
    # Target restitution 0.5 (web-sourced generic plastic-on-hard-surface
    # value; no specimen drop test of these handles). M3: set by drop tests
    # in the model (50 mm -> 0.49, 20 mm -> 0.50). M4: the same damping
    # ratio (kd ~ sqrt(mass)); its slim collar and handle make its first
    # bounce geometry-dominated (0.10-0.36 over 600-1200 N s/m, no trend).
    m3_handle_damping_n_s_m: float = 1800.
    m4_handle_damping_n_s_m: float = 830.
    m3_tool_limit_nm: float = 0.35
    m4_tool_limit_nm: float = 0.60
    # Sensor replacement and aiming are done on a secured frame: both M4s
    # start tightened (None = by task: sensor and aim).
    frame_pretightened: bool | None = None
    # Every scene starts from rest: the loose rail on its washer, the loose
    # sensor at its slot end, a pre-tightened frame in equilibrium. This much
    # physics runs at construction and the settled state is the reset state.
    initial_settle_s: float = 0.5
    fixture_mode: bool = False
    fixture_hold_sensor: bool = False
    fixture_target: str = "sensor_left"
    fixture_driver: str | None = None
    fixture_hand_driver: str | None = None
    fixture_no_driver: bool = False
    fixture_withdraw_driver: bool = False
    fixture_dual_driver: bool = False
    fixture_approach_driver: bool = False
    fixture_hand_target: str | None = None
    fixture_hand_prestart_rad: float = 0.
    fixture_hand_arm: str = "left"
    fixture_preload_test: bool = False
    task_contract: str = ""

    @property
    def frame_is_pretightened(self):
        return (self.task in PRETIGHTENED_TASKS if self.frame_pretightened is None
                else self.frame_pretightened)

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"Unknown tightening task: {self.task}")
        from .task import contract_for
        contract_for(self.task, self.task_contract)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.solver_iterations) is not int or self.solver_iterations < 1:
            raise ValueError("solver_iterations must be positive")
        if type(self.contact_capacity) is not int or self.contact_capacity < 128:
            raise ValueError("contact_capacity must be at least 128")
        for key in ("dt", "contact_gap_m", "thread_mu", "bearing_mu", "clamp_mu",
                    "m3_bearing_mu", "aluminium_bearing_mu", "sensor_face_mu", "sensor_strip_mu",
                    "bearing_contact_stiffness_n_m", "m3_bearing_contact_stiffness_n_m",
                    "front_bearing_contact_stiffness_n_m",
                    "m3_bearing_damping_n_s_m", "screw_armature_kg_m2", "screw_axial_armature_m3_kg", "screw_axial_armature_m4_kg", "bearing_damping_n_s_m",
                    "top_bearing_damping_n_s_m", "rail_seat_stiffness_n_m", "rail_seat_damping_n_s_m", "rail_lift_armature_kg",
                    "rail_pitch_stop_nm_rad", "rail_pitch_stop_damping_nm_s_rad", "rail_yaw_clearance_rad", "rail_yaw_stop_nm_rad", "rail_clamp_lever_m", "rail_rotation_damping_nm_s_rad", "m3_handle_damping_n_s_m", "m4_handle_damping_n_s_m",
                    "frame_contact_stiffness_n_m", "frame_contact_damping_n_s_m",
                    "clamp_contact_dmax", "solver_impratio", "bracket_armature_kg",
                    "bracket_play_damping_n_s_m",
                    "clutch_armature_kg_m2", "clutch_lock_stiffness_nm_rad", "screw_lock_stiffness_nm_rad",
                    "screw_axial_lock_stiffness_n_m", "slot_stop_stiffness_n_m", "slot_stop_damping_s", "screw_stick_speed_rad_s", "rail_slot_half_travel_m", "bracket_hole_clearance_m",
                    "clutch_lock_damping_nm_s_rad", "clutch_notch_rad", "clutch_climb_rad",
                    "clutch_rollover_fraction",
                    "m3_tool_limit_nm", "m4_tool_limit_nm", "rail_level_tolerance_rad"):
            x = getattr(self, key)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if isinstance(self.rail_play_damping_n_s_m, bool) or not isinstance(
                self.rail_play_damping_n_s_m, (int, float)) or not math.isfinite(
                self.rail_play_damping_n_s_m) or self.rail_play_damping_n_s_m < 0:
            raise ValueError("rail_play_damping_n_s_m must be finite and nonnegative")
        if self.frame_pretightened is not None and type(self.frame_pretightened) is not bool:
            raise ValueError("frame_pretightened must be boolean or None")
        if (isinstance(self.initial_settle_s,bool) or not isinstance(self.initial_settle_s,(int,float))
                or not 0<=self.initial_settle_s<=2):
            raise ValueError("initial_settle_s must be 0..2 s")
        if type(self.fixture_mode) is not bool:
            raise ValueError("fixture_mode must be boolean")
        if type(self.fixture_hold_sensor) is not bool or (self.fixture_hold_sensor and not self.fixture_mode):
            raise ValueError("fixture_hold_sensor requires assisted fixture_mode")
        if self.fixture_target not in ("sensor_left","sensor_right","frame_front","frame_top"):
            raise ValueError("Unknown fixture fastener")
        if self.fixture_driver not in (None,"M3_driver","M4_driver"):
            raise ValueError("Unknown fixture driver")
        if self.fixture_hand_driver not in (None,"M3_driver","M4_driver") or (self.fixture_hand_driver and not self.fixture_mode):
            raise ValueError("fixture_hand_driver requires assisted fixture_mode and a known driver")
        if self.fixture_hand_driver and self.fixture_driver:
            raise ValueError("Select a hand-staged driver or a guided fixture driver")
        if type(self.fixture_no_driver) is not bool or (self.fixture_no_driver and not self.fixture_mode):
            raise ValueError("fixture_no_driver requires assisted fixture_mode")
        if self.fixture_no_driver and (self.fixture_driver or self.fixture_hand_driver):
            raise ValueError("Driver fixture options are mutually exclusive")
        if type(self.fixture_withdraw_driver) is not bool or (self.fixture_withdraw_driver and
                (not self.fixture_mode or self.fixture_no_driver or self.fixture_hand_driver)):
            raise ValueError("A retractable guide requires an assisted guided driver")
        if type(self.fixture_dual_driver) is not bool or (self.fixture_dual_driver and
                (not self.fixture_mode or self.fixture_no_driver or self.fixture_hand_driver or
                 self.fixture_target not in ("sensor_left", "frame_top"))):
            raise ValueError("Dual guided drivers require a sensor-left or frame-top fixture")
        if self.fixture_dual_driver and self.fixture_target=="sensor_left" and not self.fixture_withdraw_driver:
            raise ValueError("M3 transfer fixture needs bit retraction before lateral travel")
        if type(self.fixture_approach_driver) is not bool or (self.fixture_approach_driver and
                (not self.fixture_mode or not self.fixture_withdraw_driver)):
            raise ValueError("Assisted axial approach needs a retractable guide")
        if self.fixture_hand_target not in (None,"frame_top","sensor_left") or (
                self.fixture_hand_target is not None and
                (not self.fixture_mode or self.fixture_hand_driver!=
                 ("M3_driver" if self.fixture_hand_target=="sensor_left" else "M4_driver"))):
            raise ValueError("Assisted hand target requires its matching free driver")
        if (isinstance(self.fixture_hand_prestart_rad,bool) or
                not isinstance(self.fixture_hand_prestart_rad,(int,float)) or
                not math.isfinite(self.fixture_hand_prestart_rad) or
                not 0<=self.fixture_hand_prestart_rad<=20 or
                (self.fixture_hand_prestart_rad and self.fixture_hand_target is None)):
            raise ValueError("Finite assisted prestart angle 0..20 rad requires a hand target")
        if self.fixture_hand_arm not in ("left","right"):
            raise ValueError("Assisted hand arm must be left or right")
        if type(self.fixture_preload_test) is not bool or self.fixture_preload_test:
            raise ValueError("The fixed-angle preload fixture is retired; drive screws with a tool")

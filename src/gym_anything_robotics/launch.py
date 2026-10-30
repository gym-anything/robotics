"""Versioned launch configuration and lightweight environment discovery."""

from copy import deepcopy
from dataclasses import dataclass, field
from importlib import metadata, util
import json
from pathlib import Path
import sys

from . import registry

SCHEMA = "gar.launch.v1"
ENVIRONMENTS = {
    "label-application": ("dual-fr3", "label_application", "1.5", "Pressure-sensitive label shell + adhesive interface with dual FR3 pack/lab/receiving benches."),
    "barcode-scanning": ("dual-fr3", "barcode_scanning", "1.5", "Handheld barcode imager (physical trigger, cradle, thin-lens imager, LEDs, sensor, zxing decoder) at stockroom shelving, a checkout counter and a library desk; dual-fr3 (default), dual-panda, single-fr3, g1. Refused as unreachable: single-fr3 with checkout-present, checkout-bulky or library-checkin; g1 with checkout-bulky. Scanning actions take about 75 s of wall time per simulated second on a quiet GPU and minutes on a shared one: give HTTP clients a timeout of at least 600 s."),
    "filleting": ("dual-fr3", "filleting", "1.5", "Fillet fish and trim meat with two FR3 arms and a flexible fillet knife held in authored knife-handle inserts: explicit FEM tissue with a knife-edge cutting law, cohesive interfaces and bones (trout, salmon side, pork tenderloin); dual-fr3 (default), dual-panda and single-fr3 for take-knife only. Slow: tens to a few hundred wall seconds per simulated second (cutting the most): give HTTP clients a timeout of at least 600 s."),
    "concrete-finishing": ("mobile-fr3-duo", "concrete_finishing", "1.5", "Finish concrete and fill joints in a basement pump room (B-012) with a Franka Mobile FR3 Duo holding real Marshalltown floats, trowels, a darby, an edger, a jointer, a sealant tool and knives: a new pump pad (darby, edge, float, trowel), a floor patch, CMU infill joints, a sealant control joint and a wallboard patch. Concrete, repair mortar, masonry mortar, sealant and compound are 2.5-D yield-stress fields that age, bleed and evaporate; a tool is carried by the paste's bearing capacity and shapes it only by contact (cut, fill, roll, spill, heave); a wait command advances the material clock."),
    "classroom-arrangement": ("mobile-fr3-duo", "classroom_arrangement", "1.5", "Rearrange a furnished 71 m2 classroom (24 Virco 785 desks, 28 9018 chairs on glides, tutoring table, teacher's desk) with a Franka Mobile FR3 Duo: glide friction (static/sliding, by glide type), tipping, base traction, FR3 reflexes."),
    "painting": ("mobile-fr3-duo", "painting", "1.5", "Repaint a US bedroom wall (roller and tray, angled sash brush and cut pail), paint acrylic on watercolour paper, spray aerosol enamel through a stencil, with a Franka Mobile FR3 Duo: yield-stress paint films that flow, level, sag, dry, soak in and lap; roller nap, bristle tufts and a finger-pressed aerosol valve."),
    "surface-rinsing": ("mobile-fr3-duo", "surface_rinsing", "1.5", "Rinse surfaces in a restaurant dish room (three-compartment pot sink with pre-rinse unit, soiled dish table with pre-rinse sink, prep sink, can-wash bay with hose reel) with a Franka Mobile FR3 Duo: contact-opened valves, momentum-carrying jets that splatter, inertial thin films on ware, pools that fill, drain and float things, surfactant, foam and soil carried by the water, soap-dependent grip friction."),
    "soil-excavation": ("mobile-fr3-duo", "soil_excavation", "1.5", "Dig soil in a front yard with a Franka Mobile FR3 Duo on the walkway: a spade, trenching shovel, trowel, hand mattock, clamshell post-hole digger, 2x4 tamper and a drill-mounted planting auger in a two-way coupled granular soil (Newton implicit MPM: layered mulch, topsoil, turf, clay-loam subsoil, hardpan, sand; cohesion lost on loosening, regained on compaction; stones, a root ball, a braced post, an irrigation lateral)."),
    "wine-pouring": ("dual-fr3", "wine_pouring", "1.5", "Robotic wine bar: GPU DFSPH wine/water (2 mm particles, two-way coupled) in real glassware (LEONOR bottle, PURE/ISO/lined glasses, carafes, dump bucket) poured by dual FR3 + Franka Hands; volumes, spills and drips from the liquid itself."),
    "percussion": ("dual-fr3", "percussion", "1.5", "Music-therapy room: dual FR3 + Franka Hands play an Orff alto xylophone with box resonators, resonator bars and a 14 in coated drum with drumsticks, rubber mallets, a felt beater or a fingertip; modal instruments (Rucz bars, air-loaded Ambassador head) struck through an implicit audio-rate vibro-contact, sound simulated at the client's ear and the station microphone."),
    "drilling": ("dual-fr3", "drilling", "1.5", "Drill holes in a building workshop: a concrete wall bay (C25/30 with seeded rebar and aggregate, SDS-plus rotary hammer, anchors, hole cleaning) and a drill-press bench (steel, aluminium, sheet and timber; cordless drill, centre punch, vise), operated only through contact by dual FR3 + Franka Hands. Holes are physical records with proxy colliders; feed follows force through published cutting laws."),
    "gift-bags": ("dual-fr3", "gift_bags", "1.5", "Open paper packaging at a boutique cash-wrap counter (factory-flat laminated rope-handle tote on a stack, bottom boards, boxed candle) a dining table (two-piece gift box with its lid air film; taped wrapped present) and a back-room receiving table (sealed corrugated carton: packing-tape seam, four creased flaps). Robots: two FR3 arms with Franka Hands by default; a Panda pair, and a single FR3 or Panda at the dining table, are the other qualified setups (no humanoid). Implicit contact world for orthotropic paper with creases, tearing, tape and rope handles."),
    "syringe-handling": ("dual-fr3", "syringe_handling", "1.5", "Handle syringes at a clinical skills-laboratory counter: Luer-lock syringes whose plungers push against a liquid network (seal stick and slip, needle and cannula resistance, trapped air, a vial's headspace, a saline lock's line and clamp, tissue that stores and gives back), needles that puncture and track with elastic deflection and a terminal yield flag, Luer joints that must be turned, sheaths and a sharps container, worked by dual FR3 + Franka Hands; doses, delivery rates, air, pressures, needle paths, touches and spills are measured from the liquid and the rigid state."),
    "ingredient-mixing": ("dual-fr3", "ingredient_mixing", "1.5", "Mix ingredients at a teaching-kitchen station: a GPU material-point mixture that carries the mass of every ingredient (dry powders, wetting, lumps, shear-thinning batters, yield-stress pastes, cold-set and cooked starch) in a steel bowl, a glass bowl or a saucepan on an induction hob, worked by dual FR3 + Franka Hands with a whisk, spatula, spoon or hand mixer; uniformity, dry flour, lumps, mixing work, heat and scorching are measured from the material itself."),
    "tighten-codex": ("dual-fr3", "tighten_codex", "1.5", "Industrial W4F sensor and 20 mm profile with native hex contact, prestarted M3/M4 helical screws and measured bearing preload."),
    "deposit-cans": ("dual-fr3", "deposit_cans", "1.5", "Documented v6/scene7 physical-component release; seed0 reference/high_fill."),
    "whiteboard": ("g1", "whiteboard", "1.5", "Mounted G1 camera; explicit task/variation options."),
    "titration": ("panda", "titrate", "1.5", "Requires explicit liquid_numerics or apparatus_fixture=True."),
    "pencap": ("dual-panda", "pencap", "1.5", "Studio baseline with fixed gripper effort."),
    "lockkey": ("panda", "lockkey", "1.5", "Pinned Panda lock/key baseline."),
    "laundry": ("dual-panda", "laundry", "1.5", "Garment selected through constructor config."),
    "football": ("g1", "football", "1.5", "CPU Newton/MuJoCo baseline; no balance policy."),
    "welding": ("ur10e-fr3", "welding", "1.5", "Robotic GMAW (MIG) cell: a UR10e carries a cobot MIG gun and an FR3 with a Franka Hand handles parts at a steel fixture table; position the torch, tack a bracket, lay a 1/4 in horizontal fillet judged by AWS D1.1 sections. Transient heat with melting in the coupon (GPU), a constant-voltage arc through a grounded circuit, a surface-tension pool, fusion, porosity and a fused-throat bond. Slow while the arc burns, and how slow depends on what else the shared GPU carries: every request adds a fixed cost, mostly the observation's images, and other jobs on the GPU slow the physics as well as the renders. A runtime measurement gates arc-on commands of 100 ms or longer against the 200x limit, and whether they stay under it depends on the load: with one camera they held under up to four other render workers; with the three policy cameras they failed under two other render workers and a dozen other GPU jobs; under five other render workers no command up to 500 ms held. The latest measured figures, with the load they ran under, are in the policy's capabilities (hardware.wall_time). Put tool_io changes in the same {'actions': [...]} request as the next motion. The render worker (about 4.2 GiB of GPU memory) stays warm 600 s after each request (renderer_options idle_release_s). gar commands use the session's declared client timeout (3600 s); give other HTTP clients at least that."),
    "forklift": ("h1-2", "forklift", "1.5", "Drive a 2.5 t LPG counterbalance forklift: a Unitree H1-2 with Inspire hands seated in the truck operates its steering wheel and knob, pedals, direction lever, mast levers, key, parking brake and horn with its own hands and feet. Super-elastic tyre model, torque-converter drivetrain, hydrostatic steering and staged mast hydraulics fitted to the Hyster H2.5A sheet; warehouse (racks, floor bays, conveyor), dock (levelers, sprung trailers with chocks or a restraint) and yard (ramp) presets; Unitree low-level motor channels at 0.5 ms; head camera at 30 Hz. Runtime, measured per robot: the policy's capabilities carry it (hardware.wall_time, the backend's one note, with the figures behind it in hardware.shortest_command_within_200x_s, wall_per_sim_per_request, camera_frame_render_s, render_worker_idle_release_s, render_worker_cold_start, render_worker_gpu_mib, session_host_memory_mib and physics_trace); none is copied here. A command carries at most 30 s of simulated time; gar commands use the session's declared client timeout (hardware.client_timeout_s); give other HTTP clients at least that. Assets: python -m forklift.assets.install and python -m robots.models.h1_2.install (gar doctor forklift checks both)."),
    "sewing": ("dual-fr3", "sewing", "1.5", "Sample-room sewing cell: two FR3 arms with Franka Hands at an industrial single-needle lockstitch machine (Juki DDL-9000C class: 40 N presser foot, drop feed, 5,000 sti/min). Load two muslin plies under the foot, sew a plain seam with backtacks, baste, pivot a corner, or sew round a rounded corner, driving the machine through its control box (tool_io). Woven plies as orthotropic shells in an implicit step with fabric compression, Coulomb friction and the air under the cloth; the presser foot a body; needle puncture, deflection and plate strikes; stitch formation, thread tension and trimming from the simulated state. Also a Panda pair, a single FR3 and a standing G1 (--robot). Runtime per deployment, as measured, is in the policy's capabilities (hardware.wall_time); a command carries at most 2 s of simulated time, and HTTP clients need a timeout of at least 900 s."),
}


def describe(environment_id, backend="newton"):
    name = registry._ALIASES.get(environment_id, environment_id)
    if (name, backend) not in registry._REGISTRY:
        raise ValueError(f"Unknown environment/backend: {environment_id}/{backend}")
    robot, package, version, notes = ENVIRONMENTS.get(name, (None, None, None, "External registration"))
    from robots.catalog import default_robot, environment_robots
    robot=default_robot(name) or robot
    return {"schema": "gar.environment.v1", "environment_id": name, "backend": backend,
            "default_robot": robot, "robots": environment_robots(name), "implementation_package": package,
            "engine_version": version, "notes": notes,
            "qualification": "Discovery metadata, not a physical acceptance result."}


@dataclass(frozen=True)
class LaunchSpec:
    environment_id: str
    backend: str = "newton"
    robot: str | dict | None = None
    backend_options: dict = field(default_factory=dict)
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA:
            raise ValueError(f"Unsupported launch schema: {self.schema}")
        if not isinstance(self.environment_id, str) or not isinstance(self.backend, str):
            raise ValueError("Environment and backend must be strings")
        describe(self.environment_id, self.backend)
        if self.robot is not None:
            from robots.catalog import select_for_backend
            name=registry._ALIASES.get(self.environment_id,self.environment_id)
            select_for_backend(name,self.robot)
            object.__setattr__(self,"robot",deepcopy(self.robot))
        if not isinstance(self.backend_options, dict):
            raise ValueError("backend_options must be an object")
        reserved = {"environment_id", "backend", "robot", "auto_reset"} & self.backend_options.keys()
        if reserved:
            raise ValueError(f"Reserved backend options: {sorted(reserved)}")
        if "device" in self.backend_options and isinstance(self.backend_options.get("config"), dict) and "device" in self.backend_options["config"]:
            raise ValueError("Device supplied at two levels; choose exactly one")
        if "attach_ready_file" in self.backend_options:
            raise ValueError("Managed launch owns a new environment; attach with gar.connect or --url")
        # Validate finite JSON and detach from caller-owned mutable dictionaries.
        encoded = json.dumps(self.backend_options, allow_nan=False)
        object.__setattr__(self, "backend_options", json.loads(encoded))
        # An environment may check its robot against the other launch options (e.g. a task that needs a hand) before
        # any session exists: optional robots.catalog ENVIRONMENTS[env]["launch_check"] = "module:function".
        from robots.catalog import ENVIRONMENTS
        check = ENVIRONMENTS.get(registry._ALIASES.get(self.environment_id, self.environment_id), {}).get("launch_check")
        if check:
            import importlib
            module, function = check.split(":")
            getattr(importlib.import_module(module), function)(deepcopy(self.robot), deepcopy(self.backend_options))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Launch configuration must be an object")
        unknown = set(value) - {"schema", "environment_id", "backend", "robot", "backend_options"}
        if unknown or "environment_id" not in value or "schema" not in value:
            raise ValueError(f"Launch requires schema/environment_id; unknown fields: {sorted(unknown)}")
        return cls(**value)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return {"schema": self.schema, "environment_id": self.environment_id,
                "backend": self.backend, "robot": deepcopy(self.robot),
                "backend_options": deepcopy(self.backend_options)}

    def create(self):
        # The whiteboard task-set adapter is chosen by its options; native launches keep their factory and
        # option handling.
        if (registry._ALIASES.get(self.environment_id, self.environment_id) == "whiteboard"
                and self.backend == "newton"
                and {"catalogue_task", "task_contract"} & self.backend_options.keys()):
            from .backends.newton_whiteboard_task_set import NewtonWhiteboardTaskSetBackend
            from .env import RoboticsEnv
            from robots.catalog import default_robot, select_for_backend
            robot = select_for_backend("whiteboard", default_robot("whiteboard") if self.robot is None else self.robot)
            return RoboticsEnv(NewtonWhiteboardTaskSetBackend(robot=robot, **deepcopy(self.backend_options)))
        return registry.create(self.environment_id, backend=self.backend, robot=self.robot,
                               **deepcopy(self.backend_options))


def doctor(spec):
    descriptor = describe(spec.environment_id, spec.backend)
    package = descriptor["implementation_package"]
    found = package is None or util.find_spec(package) is not None
    versions = {}
    for name in ("newton", "warp-lang", "mujoco"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    coordinator_ok = versions["newton"] is not None
    # an environment may declare its own dependency checks (`<package>.doctor.checks(spec)`); imports,
    # files and variables only, never a simulator
    env_checks = None
    if found and package is not None and util.find_spec(f"{package}.doctor") is not None:
        import importlib
        env_checks = importlib.import_module(f"{package}.doctor").checks(spec)
    env_ok = env_checks is None or bool(env_checks.get("ok"))
    return {"schema": "gar.doctor.v1", "ok": found and coordinator_ok and env_ok,
            "python": sys.executable, "implementation_available": found,
            "coordinator_packages": versions, "environment_checks": env_checks, "environment": descriptor,
            "configuration": spec.to_dict(),
            "scope": "Import/package availability only; no physics or renderer started.",
            "not_checked": ["assets", "GPU execution", "worker engine version", "physical qualification"]}

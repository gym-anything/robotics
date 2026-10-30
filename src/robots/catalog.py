"""Data-only discovery. No Newton, Warp, assets download or GPU initialization."""

import json
from pathlib import Path
from .spec import RobotSpec, RobotSetup, _name

ROOT = Path(__file__).resolve().parent


def model(reference, *, root=ROOT):
    if not isinstance(reference, str):
        raise TypeError("Model reference must be a string")
    name, marker, revision = reference.partition("@")
    _name(name, "model")
    files = [
        Path(root) / folder / name / "robot.json"
        for folder in ("models", "attachments")
    ]
    matches = [p for p in files if p.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous robot component: {reference}")
    result = RobotSpec.from_dict(json.loads(matches[0].read_text()))
    if result.id != name or marker and str(result.revision) != revision:
        raise ValueError(f"Unavailable robot revision: {reference}")
    return result


def resolve(value, *, root=ROOT):
    if isinstance(value, RobotSetup):
        result = value
    elif isinstance(value, dict):
        result = RobotSetup.from_dict(value)
    elif isinstance(value, (str, Path)):
        name = str(value)
        if name.endswith(".json"):
            path = Path(name).expanduser().resolve()
        else:
            _name(name, "setup")
            path = Path(root) / "rigs" / f"{name}.json"
        if not path.is_file():
            raise ValueError(f"Unknown robot setup: {value}")
        result = RobotSetup.from_dict(json.loads(path.read_text()))
    else:
        raise TypeError(
            "Robot setup must be a catalog id, JSON path, object or RobotSetup"
        )
    data = result.to_dict()
    for name, instance in data["instances"].items():
        definition = model(instance["model"], root=root).to_dict()
        if instance["controller_profile"] not in definition.get("actuation", {}).get(
            "profiles", {}
        ):
            raise ValueError(
                f"Unknown controller profile for {name}: {instance['controller_profile']}"
            )
        if instance.get("initial_pose") is not None and instance[
            "initial_pose"
        ] not in definition.get("poses", {}):
            raise ValueError(f"Unknown initial pose for {name}")
    for name, attachment in data.get("attachments", {}).items():
        definition = model(attachment["model"], root=root).to_dict()
        if definition["root"]["mounting"] != "attachment":
            raise ValueError(f"{name} is not an attachment")
        if attachment["mount_profile"] not in definition.get("support", {}).get(
            "mount_profiles", {}
        ):
            raise ValueError(f"Unknown mount profile for {name}")
    return result


def list_models():
    return [
        model(p.parent.name).to_dict()
        for folder in ("models", "attachments")
        for p in sorted((ROOT / folder).glob("*/robot.json"))
    ]


def list_setups():
    return [resolve(p).to_dict() for p in sorted((ROOT / "rigs").glob("*.json"))]


def describe(value):
    setup = resolve(value)
    data = setup.to_dict()
    return {
        "setup": data,
        "fingerprint": setup.fingerprint,
        "models": {
            item["model"]: model(item["model"]).to_dict()
            for item in data["instances"].values()
        },
        "note": "Catalog metadata; live backend capabilities and qualification remain authoritative.",
    }


# Preserve environment-specific baseline control settings behind old aliases.
# These are compatibility names, not duplicated hardware implementations.
ENVIRONMENTS = {
    "deposit-cans": {
        "default": "dual-fr3",
        "aliases": {"dual-fr3": "deposit-fr3", "dual-panda-wsg50": "deposit-panda"},
    },
    "tighten-codex": {"default": "dual-fr3", "aliases": {
        "dual-fr3": "tighten-fr3", "single-fr3": "tighten-fr3-single",
        "dual-panda": "tighten-panda"},
        "validator": "tighten_codex.kinematics:validate_setup"},
    "label-application": {
        "default": "dual-fr3",
        "aliases": {"dual-fr3": "label-fr3", "dual-panda": "label-panda", "single-fr3": "label-fr3-single"},
        "unsupported": {"g1": "the G1/Dex3 hands have no label-contact fingertip pads, pinch-load mapping, "
                              "arm IK or bench cameras qualified in label-application; the qualified rigs are dual-fr3, dual-panda and single-fr3"},
    },
    "barcode-scanning": {
        "default": "dual-fr3",
        # g1: floating G1 + Dex3 standing on PD servos (no support) at 0.5 ms; measured: stands in every preset, holds
        # the scanner by finger contact and closes its trigger with its own index finger (decoded read); out of the
        # cradle only by a tipped lift (3 of 7), never put back (measured)
        "aliases": {"dual-fr3": "scan-fr3", "dual-panda": "scan-panda", "single-fr3": "scan-fr3-single", "g1": "scan-g1"},
        "validator": "barcode_scanning.robot_setup:validate_setup",
    },
    "filleting": {
        "default": "dual-fr3",
        "aliases": {"dual-fr3": "filleting-fr3", "dual-panda": "filleting-panda", "single-fr3": "filleting-fr3-single"},
        # g1: not integrated (no measured run); payload does not decide it (the knife is 88 g)
        "unsupported": {"g1": "not integrated in filleting: the knife-handle inserts, the grip and the cutting "
                              "reactions are modelled on the Franka Hand only, and no G1 run of any filleting task "
                              "exists (the G1/Dex3 stands in barcode_scanning's rig scan-g1, but holding and steering "
                              "the 88 g knife through tissue with the Dex3 is unmeasured); the qualified rigs are "
                              "dual-fr3, dual-panda and single-fr3"},
        "validator": "filleting.robot_setup:validate_setup",
    },
    "welding": {
        # ur10e-fr3: UR10e torch arm with the MIG gun mounted by the welding environment + FR3/Franka Hand handling
        # arm; ur10e: the torch arm alone (tasks that need no hand)
        "default": "ur10e-fr3",
        # dual-fr3: the right FR3 carries a 1.6 kg light torch (RM2 + KS-2 class) inside its 3 kg payload, the hose on a
        # balancer; its wrist torques along the weld are measured
        # single-fr3: that FR3 torch arm alone (tasks that need no hand)
        # dual-panda: the same light torch on the right Panda's link8 (its Franka Hand removed: Panda profile
        # welding_torch), the left Panda a Franka Hand; g1: the G1 standing behind the table, the light torch clamped
        # to the back of its right Dex3 palm, the left Dex3 handling (welding/humanoid.py)
        "aliases": {"ur10e-fr3": "welding-ur10e-fr3", "ur10e": "welding-ur10e", "dual-fr3": "welding-fr3",
                    "single-fr3": "welding-fr3-single", "dual-panda": "welding-panda", "g1": "welding-g1"},
        "validator": "welding.robot_setup:validate_setup",
        # the robot against the task at launch time (gym_anything_robotics/launch.py): torch-only rigs refuse tack
        "launch_check": "welding.variants:check_launch",
    },
    "forklift": {
        # h1-2: floating Unitree H1-2 + Inspire hands seated in a 2.5 t counterbalance truck, driving through the
        # truck's own controls; the seated operator is built from the selected humanoid rig's catalogue model
        # (forklift/operator.py Skeleton + SEAT_FIT). The G1 seats too (rig forklift-g1) but is refused (it cannot press
        # the service brake through its travel); the arm templates are fixed-base (welded to the world) and have no limb
        # for the pedals.
        # drive-kit: the non-humanoid route: a vehicle-testing driving-robot kit (model drive_kit: steering, pedal,
        # lever, selector, hand-brake, key and button actuators with the products' published limits) bolted into the
        # cab, acting on the same controls by contact only (forklift/operator.py, kit fit).
        "default": "h1-2",
        "aliases": {"h1-2": "forklift-h1-2", "drive-kit": "forklift-drive-kit"},
        "unsupported": {
            "g1": "measured in the cab, with collisions, the G1 seated by contact (rig forklift-g1; physical probe E14g; assisted as the H1-2's E14b fixtures: pelvis held, wrist guide <= 60 N, foot guide <= 150 N): from the seat it works the lift lever by one finger (pulled 91.0 %, pushed 89.7 % of its travel, the forks up 0.44 m and down 0.63 m) and carries the spinner knob a full circle one way (the wheel turned 353 deg) and 10 deg the other way before the knob leaves its Dex3 hand (the wheel turned 50 deg), but it cannot press the service brake from where it sits: seated with its back on the backrest and the seat at its forward limit, its straight leg puts the toe on the brake pad only at rest (4.5 mm off), 30.9 mm short of it at half travel and 58.8 mm short at full travel, and in physics its foot never moved the pedal (0 % of travel). Slid 15 cm forward on the cushion, its foot pressed the pedal to 75 % of travel (934 N on the pad) only while the external support held its pelvis there against the pedal's reaction (1.5 kN at the peak), its right ankle pressing the inching pedal too (312 N) and its left ankle the hood (251 N). A G1 operator could not stop the truck from a seat it can hold by itself. The non-humanoid route is the driving-robot kit: --robot drive-kit.",
            "single-fr3": "a fixed-base arm: the catalogue FR3 is mounted fixed and the shared builder welds its base to the world frame, so it cannot ride in the truck's moving cab; and one arm cannot work at once what a driver works with two hands and two feet: the spinner knob, the cowl levers, the service brake (10-1800 N at its pad over 64 mm of travel, the master cylinder's line pressure pushing back), the inching pedal (14-60 N over 67 mm by its own spring, and the service brake's force too past 17 % of its travel, where its lug presses the brake pedal) and the accelerator (7-35 N over 50 mm) (pedal springs forklift/params.py PEDAL_*, the brake's line pressure drivetrain.pedal_force_at at params BRAKE_PAD_ARM, pad lever arms from the truck's meshes). The non-humanoid route is the driving-robot kit (--robot drive-kit): purpose-built actuators on each control",
            "dual-fr3": "two fixed-base arms: the catalogue FR3s are mounted fixed and the shared builder welds each base to the world frame, so neither can ride in the truck's moving cab; mounted at the seat, two hands could steer and work the levers but leave no limb for the three pedals a driver works continuously while steering and lifting: service brake 10-1800 N at its pad over 64 mm (the master cylinder's line pressure pushing back), inching pedal 14-60 N over 67 mm by its own spring and the brake's force past 17 % of its travel, accelerator 7-35 N over 50 mm. The non-humanoid route is the driving-robot kit (--robot drive-kit)",
            "dual-panda": "as dual-fr3: the catalogue Pandas are mounted fixed (bases welded to the world frame, so they cannot ride in the moving cab), and two hands leave no limb for the service brake (10-1800 N at its pad over 64 mm, the master cylinder's line pressure pushing back), the inching pedal (14-60 N over 67 mm by its own spring, and the brake's force past 17 % of its travel) and the accelerator (7-35 N over 50 mm) while steering and lifting. The non-humanoid route is the driving-robot kit (--robot drive-kit)",
        },
        "validator": "forklift.robot_setup:validate_setup",
    },
    "sewing": {
        # dual-fr3: two FR3 + Franka Hands on one pedestal in the operator's seat; dual-panda
        # the same hands on Pandas; single-fr3 the right arm alone (tasks that one hand can do: sewing/variants.py);
        # g1: the Unitree G1 with Dex3 hands standing at the table (sewing/g1.py)
        "default": "dual-fr3",
        "aliases": {"dual-fr3": "sewing-fr3", "dual-panda": "sewing-panda", "single-fr3": "sewing-fr3-single",
                    "g1": "sewing-g1"},
        "validator": "sewing.robot_setup:validate_setup",
    },
    "classroom-arrangement": {
        "default": "mobile-fr3-duo",
        # Franka Mobile FR3 Duo (TMR base + Spine + two FR3 with Franka Hands) and the same platform with one arm
        "aliases": {"mobile-fr3-duo": "classroom-mobile-fr3-duo", "mobile-fr3-single": "classroom-mobile-fr3-single"},
        "validator": "classroom_arrangement.robot_setup:validate",
        "unsupported": {
            "g1": "no qualified locomotion controller for the floating G1 in this room: Unitree's pre-trained unitree_rl_gym G1 "
                  "walking policy has not been tried or qualified for walking among furniture over a 71 m2 room or for "
                  "pushing and lifting furniture while walking",
            "dual-fr3": "fixed-base arms cannot reach furniture across a 7.4 x 9.6 m room; use mobile-fr3-duo",
            "dual-panda": "fixed-base arms cannot reach furniture across the room; no Panda mobile base exists",
            "single-fr3": "fixed-base arm cannot reach furniture across the room; use mobile-fr3-single"},
    },
    "painting": {
        "default": "mobile-fr3-duo",
        # the Mobile FR3 Duo stack qualified in classroom_arrangement (controller profile "classroom", reflexes, base,
        # Spine, sensors) reused unchanged; painting qualifies its tool holds
        "aliases": {"mobile-fr3-duo": "painting-mobile-fr3-duo", "mobile-fr3-single": "painting-mobile-fr3-single", "g1": "painting-g1"},
        "validator": "painting.robot_setup:validate",
        "unsupported": {
            # per preset: the paint system is coupled to classroom_arrangement's Mobile
            # FR3 Duo world (base controller, Spine, joint servos, reflexes); other stacks are not integrated with it
            "dual-fr3": "wall preset: a pedestal cannot reach a 2.44 m wall from floor to ceiling (the stack needs the "
                        "mobile base and Spine); craft and spray presets are table work a pedestal pair could do, but the "
                        "scene and start grasps are built on the mobile base's floor_start mount and no pedestal layout is "
                        "built or qualified (use mobile-fr3-duo)",
            "single-fr3": "wall preset: a fixed-base arm cannot reach the wall; craft: no pedestal layout is built (use "
                          "mobile-fr3-single); spray needs two hands (hold the can, press its button)",
            "dual-panda": "wall preset: fixed-base arms cannot reach the wall and no Panda mobile base exists; craft and "
                          "spray: no pedestal layout is built or qualified"},
        # task-level limits of a qualified robot (painting.robot_setup.check_task, enforced by the backend and the env)
        "task_limits": {"g1": {"wall": "standing craft layout has no qualified full-wall reach or balance under roller load",
                                      "spray": "Dex3 valve pressing and two-hand can handling are not qualified"},
                        "mobile-fr3-single": {"spray": "the spray tasks hold the can in one hand and press its button "
                                                       "with a fingertip of the other; a single Franka Hand cannot do both"}},
    },
    "surface-rinsing": {
        "default": "mobile-fr3-duo",
        # the Mobile FR3 Duo stack qualified in classroom_arrangement (controller profile "classroom", reflexes, base,
        # Spine, sensors) with painting's contact refinements, reused unchanged
        "aliases": {"mobile-fr3-duo": "rinsing-mobile-fr3-duo", "mobile-fr3-single": "rinsing-mobile-fr3-single"},
        "validator": "surface_rinsing.robot_setup:validate",
        "unsupported": {
            "g1": "the stations are real-height fixtures with their valves on the wall behind a 0.6-0.76 m deep sink: "
                  "standing at the front edge the G1's palm stays 0.15 m short of the pot-sink faucet valve and 0.09 m "
                  "short of the pre-rinse hot blade, and a 40 deg bow from the hips still leaves 0.09 m and 0.03 m (and "
                  "the pan's wall shelf and the floor of the 32 gal can out of reach) (URDF forward kinematics over its "
                  "joint limits); no bowed or "
                  "leaning posture is qualified for the G1 in this repository, its Dex3 fingers (1.4-2.45 N m) are "
                  "marginal for the 13-22 N squeeze levers, and the water system is coupled to the Mobile FR3 Duo "
                  "world; not built or qualified",
            "dual-fr3": "a pedestal pair at one sink cannot reach the other stations, and no pedestal layout is built or "
                        "qualified (use mobile-fr3-duo)",
            "single-fr3": "fixed-base arm: no pedestal layout is built (use mobile-fr3-single)",
            "dual-panda": "no Panda mobile base exists and no pedestal layout is built or qualified"},
        "task_limits": {"mobile-fr3-single": {
            "rinse-garbage-can": "emptying the can (up to 12-18 kg with rinse water, ~60 N at the rim to tip it) needs two "
                                 "hands or more than one FR3's 3 kg payload"}},
    },
    "soil-excavation": {
        "default": "mobile-fr3-duo",
        # the Mobile FR3 Duo stack qualified in classroom_arrangement (controller profile "classroom", reflexes, base,
        # Spine, sensors) with the tool-holding additions first qualified in painting (frozen in soil_excavation/holding.py)
        "aliases": {"mobile-fr3-duo": "soil-mobile-fr3-duo", "mobile-fr3-single": "soil-mobile-fr3-single"},
        "validator": "soil_excavation.robot_setup:validate",
        "unsupported": {
            "g1": "the work is at grade: standing on the walkway the G1's lowest palm is 0.60 m above grade (URDF forward "
                  "kinematics over its joint limits); only a near-full flat-footed "
                  "squat (pelvis 0.15 m off the walkway) brings a palm to grade, and no squat, kneel or crouch posture is "
                  "qualified for the G1 in this repository (percussion and barcode_scanning qualify it standing). The "
                  "long-handled tools also exceed its hand: a Dex3-1 holds about 0.5 kg of a 5 cm object (filleting catalog "
                  "note) against a 1.9 kg spade, a 3.2 kg post-hole digger or a 2.9 kg drill with auger",
            "dual-fr3": "the yard work is on the ground 0.5-1.2 m beyond a walkway: a pedestal pair cannot reach grade "
                        "from beside the bed and has no base to follow a 1.5 m trench; no pedestal layout is built",
            "single-fr3": "fixed-base arm: no reach to grade from the walkway and no base (use mobile-fr3-single for the "
                          "one-handed tasks)",
            "dual-panda": "fixed-base arms and no Panda mobile base; no pedestal layout is built"},
    },
    "wine-pouring": {
        "default": "dual-fr3",
        # the FR3/Panda joint servos and Franka Hand qualified in barcode_scanning (controller profile reused unchanged)
        "aliases": {"dual-fr3": "pour-fr3", "single-fr3": "pour-fr3-single", "dual-panda": "pour-panda",
                    "single-panda": "pour-panda-single"},
        "validator": "wine_pouring.robot_setup:validate",
        "unsupported": {"g1": "the G1's shoulder is at the 1.09 m bar-top height and its palm 0.42 m from it (URDF FK); "
                              "a pour into a glass at a delivery spot needs the hand 0.34 m above and 0.6 m in front of "
                              "the shoulder; nor does its Dex3-1 hand have a qualified grasp of a 1.24 kg 75 mm bottle"},
    },
    "percussion": {
        "default": "dual-fr3",
        # FR3/Panda joint servos and Franka Hand qualified in barcode_scanning (controller profile reused unchanged);
        # percussion adds libfranka's acceleration/jerk limits to the command path.
        # g1: floating G1 + Dex3 standing on PD servos at 0.5 ms with arm gravity feed-forward (percussion/g1.py)
        "aliases": {"dual-fr3": "percussion-fr3", "single-fr3": "percussion-fr3-single", "dual-panda": "percussion-panda",
                    "g1": "percussion-g1"},
        "validator": "percussion.robot_setup:validate",
        "unsupported": {
            "mobile-fr3-duo": "the station mounts its arms on a parked platform at fixed instrument heights (xylophone "
                              "0.82 m, drumhead 0.74 m, resonator bars on the table) and its policy cameras on that "
                              "platform; the Mobile FR3 Duo's TMR base and Spine are qualified only in classroom_arrangement "
                              "and painting, not for strokes on these instruments; use "
                              "dual-fr3",
            "mobile-fr3-single": "the station's fixed mounts and instrument heights (see mobile-fr3-duo); use single-fr3"},
    },
    "laundry": {"default": "dual-panda", "aliases": {"dual-panda": "laundry-panda"}},
    "pencap": {"default": "dual-panda", "aliases": {"dual-panda": "pencap-panda"}},
    "lockkey": {"default": "panda", "aliases": {"panda": "lockkey-panda"}},
    "titration": {"default": "panda", "aliases": {"panda": "titration-panda"}},
    "whiteboard": {"default": "g1", "aliases": {"g1": "whiteboard-g1"}},
    "football": {"default": "g1", "aliases": {"g1": "football-g1"}},
    "gift-bags": {
        "default": "dual-fr3",
        # two FR3 / Panda arms with Franka Hands on pedestals at a counter/table;
        # the arms reuse the label_application controller profile (force-mode Franka Hand, paper-handling gains).
        # single-fr3 reaches every first interaction of the home preset but not the counter's bag stack; its backend
        # rejects the counter preset
        "aliases": {"dual-fr3": "gift-fr3", "dual-panda": "gift-panda", "single-fr3": "gift-fr3-single",
                    "single-panda": "gift-panda-single"},
        # any setup with the content of a qualified one is accepted (the label-fr3 rig is gift-fr3 under another id)
        "validator": "gift_bags.robot_setup:validate",
        # g1: the rig gift-g1 and gift_bags/g1.py exist (the floating G1 with Dex3-1 hands stands hands-off on its PD
        # servos at the counter and the dining table), but it is NOT a robot choice: see the reason (measured)
        "unsupported": {"g1": "the Unitree G1 only stands here: it has no balance controller, and moving its right hand "
                              "from the ready pose toward the gift-box lid tips it onto the floor (five 5 cm steps, "
                              "the tilt grows from 0.3 to 2.9 deg, the sixth step is refused as unreachable, and two "
                              "seconds later the pelvis is 0.06 m above the floor at 90 deg); its hands start below the table top, a 5 mm downward hand move is "
                              "refused, and no Dex3 hand has touched a packaging item. Use dual-fr3, dual-panda, "
                              "single-fr3 or single-panda",
                        "mobile-fr3-duo": "the Mobile FR3 Duo is not laid out at the gift-bags counter/table heights"},
    },
    "drilling": {
        "default": "dual-fr3",
        # two FR3 (or Panda) arms with Franka Hands on the docked tool cart, or the right arm alone for the three
        # drill-press tasks. The arm joint servos are those qualified in barcode_scanning. The Franka Hand is the same
        # hand on both arm models and gets the same controller on both (finger gain 3500 N/m, 35 N per finger = its
        # 70 N grasp force): the FR3 rigs take it from the FR3's barcode_scanning profile unchanged, the Panda rigs
        # from the Panda's `drilling` profile (the Panda's barcode_scanning arm joints unchanged, the FR3 hand's
        # finger joints; the Panda's own barcode_scanning profile limits the hand to 20 N per finger).
        "aliases": {"dual-fr3": "drill-fr3", "dual-panda": "drill-panda", "single-fr3": "drill-fr3-single",
                    "single-panda": "drill-panda-single"},
        "validator": "drilling.robot_setup:validate",
        "unsupported": {"g1": "the G1's wrist pitch and yaw joints are rated 5 N m (effort_limit in "
                              "robots/models/g1/robot.json) and its Dex3-1 hand holds at most 500 g of a 5 cm object "
                              "(Unitree's Dex3-1 product page, https://www.unitree.com/Dex3-1/, read 2026-10-07), against 1.7-3.1 kg "
                              "drills and 50-150 N drilling pushes",
                        "mobile-fr3-duo": "the Mobile FR3 Duo is not laid out or qualified at the drilling stations: the "
                                          "layouts put the arms on the docked cart's fixed mounts, and how the base's "
                                          "passive rocker suspension takes a 50-150 N drilling push has not been "
                                          "measured (planned as an optional wall override after that measurement)",
                        "mobile-fr3-single": "the Mobile FR3 with one arm is not laid out or qualified at the drilling "
                                             "stations: the layouts put the arm on the docked cart's fixed right mount, "
                                             "and how the base's passive rocker suspension takes a 50-150 N drilling "
                                             "push has not been measured (the same base as the Mobile FR3 Duo); the "
                                             "one-arm configurations qualified here are single-fr3 and single-panda",
                        "panda": "in drilling one Panda arm is the choice single-panda (rig drill-panda-single: the "
                                 "right arm at the drill-press bench, for punch-layout, press-drill and press-blind) "
                                 "and two Panda arms are dual-panda; the bare name panda is not offered because it "
                                 "does not say which",
                        "dual-panda-wsg50": "drilling's finger pads, grasps and force-mode grip are built and measured "
                                            "for the Franka Hand only (the pads are attached from that hand's "
                                            "catalogue data, drilling/scene.py; no WSG-50 has held a drilling tool "
                                            "here); use dual-panda",
                        "yam": "the YAM arm is not a catalogue robot model (robots/models has the FR3, Panda, G1 and "
                               "the Mobile FR3 platforms); drilling builds its robots from the catalogue"},
    },
    "syringe-handling": {
        "default": "dual-fr3",
        # two FR3 or two Panda arms with Franka Hands on the cart docked at a clinical skills-lab counter (the
        # teaching kitchen's cart and mounts); the right arm alone for the one task one hand can do. What stands
        # behind each offer: syringe_handling/robot_setup.py
        "aliases": {"dual-fr3": "syringe-fr3", "dual-panda": "syringe-panda", "single-fr3": "syringe-fr3-single", "g1": "syringe-g1"},
        "validator": "syringe_handling.robot_setup:validate",
        "unsupported": {"mobile-fr3-duo": "the Mobile FR3 Duo is not laid out at this counter: the bay puts the arms on "
                                          "the docked cart's fixed mounts and has no floor plan for a base",
                        "mobile-fr3-single": "the Mobile FR3 is not laid out at this counter: the bay puts the arms on "
                                             "the docked cart's fixed mounts and has no floor plan for a base"},
    },
    "ingredient-mixing": {
        "default": "dual-fr3",
        # two FR3 or two Panda arms with Franka Hands on a cart docked at a kitchen counter; the right arm alone for the
        # tasks worked with a tool from the crock in a bowl nobody holds. The left FR3 alone (rig mix-fr3-single-left)
        # builds and is offered for no task, so it is not among the choices: it is refused by name with its reason
        # (`unsupported` below; ingredient_mixing/robot_setup.py check_task). The FR3 rigs take the arm servos and the
        # hand from the FR3's barcode_scanning profile unchanged; the Panda rigs take them from the Panda's `drilling`
        # profile (the Panda's barcode_scanning arm joints, the FR3 hand's finger controller: 35 N per finger; the
        # Panda's own barcode_scanning profile limits the hand to 20 N per finger and the first grasp of this
        # environment's rollout is refused with it).
        "aliases": {"dual-fr3": "mix-fr3", "single-fr3": "mix-fr3-single",
                    "dual-panda": "mix-panda", "single-panda": "mix-panda-single"},
        "validator": "ingredient_mixing.robot_setup:validate",
        # g1: the rig mix-g1 exists (the floating G1 with Dex3-1 hands in
        # the task's own scene, the arm cart left out), but it is NOT a robot choice: see the reason (measured)
        "unsupported": {"g1": "measured at this counter without the mixture and not built into the environment "
                              "(the G1 with Dex3-1 hands, rig mix-g1, no balance controller). It stands hands-off for 30 s 0.22 m in "
                              "front of the counter's edge (tilt 0.2 deg); at 0.14 m its hanging hands start inside the "
                              "cabinet, which throws it back 46 mm (0.6 to 2.3 kN on its fingers, tilt 3.9 deg) before "
                              "it stands again. From 0.22 m a palm reaches the "
                              "vessel's near rim with the arm alone and the vessel's centre only with the trunk leaning "
                              "(the arm alone is 80 mm short); it stays up there (tilt 1.6 deg), the palm 25 mm from "
                              "where it was sent. Of the three tool handles in the crock the whisk's is within the right "
                              "arm's reach by inverse kinematics (driven there the palm stopped 23 mm short), the "
                              "spoon's needs the trunk leaning (driven there with the waist at its roll limit the palm "
                              "ended 98 mm away) and the spatula's is out of reach (11 mm short even with the trunk). A "
                              "spatula or a whisk laid in the Dex3-1 hand (assisted) is held through 1 to 6 N at its "
                              "working end and 0.2 m/s strokes over the vessel with the wrist pitch and yaw joints at "
                              "no more than 3.0 and 1.3 of their 5 N m: the wrist ratings do not stop it. But the tool "
                              "turns in the hand (up to 15 deg under those loads; the spatula 50 deg in strokes "
                              "against 6 N over the vessel), the 9.5 mm spoon handle is thinner than the 12 mm at which "
                              "the closed fingers stand off the palm and its case gave no measurement (the run was "
                              "stopped by the speed guard: spoon 24 m/s, 606 rad/s), no hand has taken a tool out of "
                              "the crock, and with no balance controller 32 to 45 N on a hand (its fingers meeting the "
                              "counter's front) put the robot on the floor, and a reach for the spatula's handle from "
                              "0.14 m ended with the "
                              "robot on the floor. The environment's arm controller, reflexes, load model and policy "
                              "interface are the Franka arms' (ingredient_mixing/world.py, the backend): a 43-joint "
                              "interface, a Dex3 grasp command and a head camera are not built. Use dual-fr3, "
                              "dual-panda, single-fr3 or single-panda",
                        "mobile-fr3-duo": "the Mobile FR3 Duo is built in classroom-arrangement and sweeping as one "
                                          "floating-base instance on a floor mount (rig classroom-mobile-fr3-duo: mount "
                                          "floor_start, profile classroom), driven over the floor by a swerve base "
                                          "controller (classroom_arrangement/base.py, action base_twist) with its arms "
                                          "on a Spine lift. This station has none of that: ingredient_mixing/scene.py "
                                          "offers two fixed mounts on the docked cart's deck (left_arm_mount, "
                                          "right_arm_mount) and reads each instance's `arm` and `gripper` joint groups, "
                                          "which the platform's model does not have (it has base_steer, base_drive, "
                                          "spine, left_arm, right_arm); the cart's deck, posts and shelf stand as fixed "
                                          "bodies where a base would have to park; ingredient_mixing/world.py has no "
                                          "base, wheel or Spine controller and the backend no base_twist action; and "
                                          "the platform's reach over the 0.914 m counter from the floor in front of it "
                                          "has not been measured. Use dual-fr3 or dual-panda",
                        "mobile-fr3-single": "the Mobile FR3 with one arm is the Mobile FR3 Duo's base and Spine with "
                                             "the right arm only (rig classroom-mobile-fr3-single); it is not built at "
                                             "this station for the reasons given for mobile-fr3-duo: no floor mount, "
                                             "the docked cart in its place, no base or Spine controller, no base_twist "
                                             "action, reach over the counter not measured. The one-arm configurations "
                                             "offered here are single-fr3 and single-panda",
                        "panda": "one Panda arm is the choice single-panda (rig mix-panda-single, the right arm) and "
                                 "two are dual-panda; the bare name panda is not offered because it does not say which",
                        "dual-panda-wsg50": "the grasps of utensil handles, the pan's handle, the mixer's handle and "
                                            "the knob are built for the Franka Hand only (its pads come from that "
                                            "hand's catalogue data, ingredient_mixing/scene.py; no WSG-50 has held a "
                                            "tool here); use dual-panda",
                        "yam": "the YAM arm is not a catalogue robot model (robots/models has the FR3, the Panda, the "
                               "G1 and the Mobile FR3 platforms); ingredient-mixing builds its robots from the catalogue",
                        "single-fr3-left": "the left FR3 alone (rig mix-fr3-single-left) builds at this station and is "
                                           "offered for no task, so it is not a robot choice. The hand mixer stands on "
                                           "its side of the vessel, but that task needs a second hand: the mixer is "
                                           "held in the batter by one hand while the other slides its switch on, up "
                                           "and off; the left hand alone can slide the switch only while the mixer "
                                           "stands on the counter, turning beaters would then be carried into and out "
                                           "of the batter, and no one-armed rollout has been run. The tools of every "
                                           "other task stand in the crock 0.37 m to the right of the vessel, 0.73 to "
                                           "0.93 m from the left arm's base, and the left hand can take only the "
                                           "nearest of the three. The rig and the "
                                           "reason per task are kept in ingredient_mixing/robot_setup.py (NOT_OFFERED, "
                                           "LEFT_TASKS, check_task) for the day a rollout stands behind an offer. Use "
                                           "single-fr3 or single-panda (the right arm alone: dry-whisk, sauce-blend, "
                                           "muffin-fold, pancake-whisk, pudding-whisk), dual-fr3 or dual-panda"},
    },
    "concrete-finishing": {
        "default": "mobile-fr3-duo",
        # Franka Mobile FR3 Duo as qualified in classroom_arrangement (same model revision and controller profile),
        # working floats, trowels, edgers, jointers, sealant tools and knives in its Franka Hands; and the same platform
        # with one arm
        "aliases": {"mobile-fr3-duo": "finish-mobile-fr3-duo", "mobile-fr3-single": "finish-mobile-fr3-single"},
        "validator": "concrete_finishing.robot_setup:validate",
        "unsupported": {
            "g1": "not offered: the pad and the floor patch are worked at floor level (pad top 89 mm), which needs "
                  "kneeling or a deep squat, and no catalog humanoid controller is qualified for either (the only "
                  "pre-trained G1 controller is Unitree's walking policy). The pad's tools are at or "
                  "over the Dex3-1 hand's 500 g payload (Unitree's Dex3-1 page, as cited in drilling's entry): darby "
                  "1.30 kg, magnesium float 0.50 kg, steel trowel 0.45 kg (concrete_finishing/params.py), and a trowel's "
                  "86 N of interface drag 70 mm below its handle is 6.0 N m about the wrist, over the G1's 5 N m wrist "
                  "pitch and yaw rating (robots/models/g1/robot.json). The wall "
                  "tasks' tools (0.03-0.16 kg) are within the hand's payload, but a tool pressed into a wall pushes a "
                  "floating G1 off balance with no balance controller qualified, and this environment's rigid world, "
                  "commands and render export are the mobile platform's (classroom_arrangement's world). Use mobile-fr3-duo",
            "dual-fr3": "not offered: fixed bases cannot stand around the 1.22 x 0.91 m pad (its far half needs the "
                        "other side) or reach the walls' 0.45-2.00 m from one place; the Mobile FR3 Duo reaches "
                        "0.65-1.10 m ahead at the floor and 0.4-2.2 m up a wall with its lift. Use mobile-fr3-duo",
            "dual-panda": "not offered for the same reason as dual-fr3 (fixed bases). Use mobile-fr3-duo",
            "single-fr3": "not offered: a fixed arm cannot reach the pad, the patch and the walls. Use mobile-fr3-single",
            "single-panda": "not offered: a fixed arm cannot reach the pad, the patch and the walls. Use "
                            "mobile-fr3-single",
        },
    },
}


def default_robot(environment):
    return ENVIRONMENTS.get(environment, {}).get("default")


def environment_robots(environment):
    item = ENVIRONMENTS.get(environment)
    if item is None:
        return []
    return [
        {
            "id": alias,
            "setup": setup,
            "default": alias == item["default"],
            "scope": "Existing Newton 1.4 YAM worker retained"
            if setup is None
            else "See resolved robot setup and environment qualification",
        }
        for alias, setup in item["aliases"].items()
    ]


def for_environment(environment, selected=None):
    """Resolve old aliases without changing the baseline's controller profile."""
    item = ENVIRONMENTS.get(environment)
    if item is None:
        raise ValueError(f"No robot deployment for {environment}")
    selected = item["default"] if selected is None else selected
    _reject_unsupported(environment, item, selected)
    name = (
        item["aliases"].get(selected, selected)
        if isinstance(selected, str)
        else selected
    )
    if name is None:
        return None  # Explicit legacy worker boundary, not a fake new adapter.
    return resolve(name)


def _reject_unsupported(environment, item, selected):
    """An environment may name catalog robots it deliberately does not support, with the reason."""
    if isinstance(selected, str) and selected in item.get("unsupported", {}):
        raise ValueError(f"{environment} does not support robot {selected!r}: {item['unsupported'][selected]}")


def select_for_backend(environment, selected):
    if environment not in ENVIRONMENTS:
        return selected
    _reject_unsupported(environment, ENVIRONMENTS[environment], selected)
    if environment == "deposit-cans":
        from .deployment import deposit_setup

        setup = deposit_setup(selected)
        return (
            selected
            if isinstance(selected, str)
            and selected in ENVIRONMENTS[environment]["aliases"]
            else setup.to_dict()
        )
    item = ENVIRONMENTS[environment]
    if isinstance(selected, str) and selected in item["aliases"]:
        return selected
    if "validator" in item:
        # The environment validates any catalog setup it can physically host.
        import importlib
        module, function = item["validator"].split(":")
        return getattr(importlib.import_module(module), function)(selected)
    if item["aliases"][item["default"]] is None:
        raise ValueError("Legacy YAM worker supports its yam baseline only")
    setup = for_environment(environment, selected)
    expected = resolve(item["aliases"][item["default"]])
    if setup.fingerprint != expected.fingerprint:
        raise ValueError(
            f"{environment} currently supports its extracted baseline setup only; see gar inspect {environment}"
        )
    return item["default"]


def describe_component(value):
    if isinstance(value, str) and any(
        (ROOT / folder / value.split("@")[0] / "robot.json").is_file()
        for folder in ("models", "attachments")
    ):
        spec = model(value)
        return {"model": spec.to_dict(), "fingerprint": spec.fingerprint}
    return describe(value)

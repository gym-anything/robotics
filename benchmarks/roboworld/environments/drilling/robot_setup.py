"""Robot-setup validator for drilling (robots.catalog ENVIRONMENTS["drilling"]["validator"]).

Accepts a catalog alias, a gar.robot-setup.v1 object or a RobotSetup when it describes one of the qualified
configurations (two FR3 arms, two Panda arms, or one FR3 or one Panda on the right mount; each with a Franka Hand on
its flange and the controller profile of PROFILES, mounted on the drilling cart), and returns that configuration's
alias. The hand is the same Franka Hand on both arm models, so its controller is the same on both: finger gain
3500 N/m, 35 N per finger (the hand's 70 N continuous grasp force, drilling/params.py FINGER_EFFORT_N). The FR3's
barcode_scanning profile carries those values; the Panda's barcode_scanning profile limits the hand to 20 N per finger,
so the Panda rigs use the Panda's `drilling` profile: its arm joints are the barcode_scanning ones unchanged and its
finger joints are the FR3 hand's (robots/models/panda/robot.json). Anything else is refused with its physical or
qualification reason (unified contract: no silent substitution). A single arm is qualified for the bench tasks whose
thrust the drill press carries (punch-layout, press-drill, press-blind); the wall preset's 3.1 kg rotary hammer exceeds
one arm's 2.27 kg tool payload."""

QUALIFIED = {"dual-fr3": "drill-fr3", "dual-panda": "drill-panda", "single-fr3": "drill-fr3-single",
             "single-panda": "drill-panda-single"}
SINGLE = {"single-fr3": "FR3", "single-panda": "Panda"}       # one arm on the right mount: alias -> arm name
MODELS = ("fr3@1", "panda@1")
HAND = "franka-hand@1"
PROFILES = {"fr3@1": "barcode_scanning", "panda@1": "drilling"}        # per arm model: the profile with the one hand
HAND_CONTROL = ("the arm servos are those qualified in barcode_scanning and the Franka Hand's controller is the same on "
                "both arm models (finger gain 3500 N/m, 35 N per finger: its 70 N grasp force): profile "
                "'barcode_scanning' of fr3@1 and profile 'drilling' of panda@1 (the Panda's own 'barcode_scanning' "
                "profile limits the hand to 20 N per finger)")
SINGLE_TASKS = ("punch-layout", "press-drill", "press-blind")
REPORT = 'drilling/REPORT.pdf, robot configurations table in section "Contact, solver and the robot"'
HAND_ONLY = (f"drilling's finger pads, grasps and force-mode grip are built and measured for the Franka Hand ({HAND}) "
             "only (the pads are attached from that hand's catalogue data, drilling/scene.py; no other gripper has "
             "held a drilling tool here)")


def _content(d):
    return {k: v for k, v in d.items() if k not in ("id", "description", "notes")}


def validate(selected):
    """Return the qualified alias for `selected` or raise ValueError with what the setup would need."""
    from robots.catalog import ENVIRONMENTS, resolve
    item = ENVIRONMENTS["drilling"]
    if isinstance(selected, str):
        if selected in item.get("unsupported", {}):
            raise ValueError(f"drilling does not support robot {selected!r}: {item['unsupported'][selected]}")
        if selected in QUALIFIED:
            return selected
        for alias, rig in QUALIFIED.items():
            if selected == rig:
                return alias
    raw = selected if isinstance(selected, dict) else (selected.to_dict() if hasattr(selected, "to_dict") else None)
    if isinstance(raw, dict):                     # an end-effector swap is refused before the catalogue resolves it
        other = sorted({str(a.get("model")) for a in (raw.get("attachments") or {}).values()
                        if isinstance(a, dict) and a.get("model") != HAND})
        if other:
            raise ValueError(f"drilling does not support end effector(s) {', '.join(other)}: {HAND_ONLY}. "
                             f"Aliases: {', '.join(QUALIFIED)}")
        insts = [i for i in (raw.get("instances") or {}).values() if isinstance(i, dict)]
        arms = sorted({str(i.get("model")) for i in insts} - set(MODELS))
        if arms:
            raise ValueError(f"drilling does not support arm model(s) {', '.join(arms)}: mounts, ready pose, reflexes "
                             f"and finger pads are qualified for {' and '.join(MODELS)} only ({REPORT}). "
                             f"Aliases: {', '.join(QUALIFIED)}")
        profiles = sorted({f"{i.get('controller_profile')!r} for {i.get('model')}" for i in insts
                           if i.get("controller_profile") != PROFILES.get(i.get("model"))})
        if profiles and len({i.get("model") for i in insts}) == 1:    # a mixed pair is told so below, not this
            raise ValueError(f"drilling does not support controller profile {', '.join(profiles)}: {HAND_CONTROL} "
                             f"({REPORT}). Aliases: {', '.join(QUALIFIED)}")
    try:
        setup = resolve(selected)
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
        if isinstance(selected, str):
            refused = ", ".join(sorted(item.get("unsupported", {})))
            raise ValueError(f"drilling has no robot setup {selected!r}: it builds the qualified rigs "
                             f"{', '.join(f'{a} ({r})' for a, r in QUALIFIED.items())} or an equal gar.robot-setup.v1 "
                             f"object ({REPORT}); refused with stated reasons: {refused}") from exc
        raise ValueError(f"Invalid robot setup for drilling: {exc}") from exc
    d = setup.to_dict()
    for alias, rig in QUALIFIED.items():
        if _content(resolve(rig).to_dict()) == _content(d):
            return alias
    inst, att = d.get("instances", {}), d.get("attachments", {})
    problems = []
    if not 1 <= len(inst) <= 2 or set(inst) - {"left", "right"}:
        problems.append("instances named 'left' and 'right' (or a single 'right')")
    if len(inst) == 1 and "right" not in inst:
        problems.append("the single arm on the right mount (the drill press is laid out for it)")
    models = {i.get("model") for i in inst.values()}
    if models - set(MODELS) or len(models) > 1:
        problems.append(f"arms of one model from {MODELS} (got {sorted(map(str, models))}): mounts, ready pose, reflexes "
                        "and finger pads are qualified for the Franka family only")
    for name, i in inst.items():
        if i.get("mount") != f"{name}_arm_mount":
            problems.append(f"{name} arm on mount {name}_arm_mount (got {i.get('mount')!r})")
        if i.get("controller_profile") != PROFILES.get(i.get("model")):
            problems.append(f"{name}: controller profile {PROFILES.get(i.get('model'))!r} ({HAND_CONTROL})")
        hands = [a for a in att.values() if a.get("parent") == f"{name}/flange"]
        if len(hands) != 1 or hands[0].get("model") != HAND or hands[0].get("mount_profile") != "franka_native":
            problems.append(f"{name}: one {HAND} on {name}/flange with mount profile franka_native ({HAND_ONLY})")
    if len(att) != len(inst):
        problems.append("exactly one Franka Hand per arm and no other attachments")
    if not problems:
        problems.append("a configuration equal to one of the qualified rigs " + ", ".join(QUALIFIED.values()))
    raise ValueError("drilling qualifies two FR3, two Panda, or one FR3 or one Panda (right) with Franka Hands; this "
                     f"setup needs: {'; '.join(problems)}. Aliases: {', '.join(QUALIFIED)} ({REPORT})")


def check_task(alias, preset, task):
    """A single arm (FR3 or Panda: the same 3 kg payload, 0.73 kg hand and 87/12 N m joints) is refused where one arm
    cannot do the work."""
    if alias in SINGLE and task not in SINGLE_TASKS:
        raise ValueError(f"drilling task {preset}/{task} needs two arms: "
                         + (f"the 3.1 kg rotary hammer exceeds one {SINGLE[alias]}'s 2.27 kg tool payload"
                            if preset == "wall" else
                            "one Franka Hand pinch does not react a hand drill's cutting torque: in the robot's "
                            "one-hand timber bore the drill turned in the pinch and the hole came out of square, "
                            "and take-drill asks for both hands")
                         + f" ({REPORT})"
                         + f"; {alias} is qualified for {', '.join(SINGLE_TASKS)}")


def check_launch(robot=None, backend_options=None):
    """Everything about a robot choice that can be decided before a session exists: the robot itself (validate) and
    whether that robot is offered for the task the launch names (check_task). Returns the qualified alias.

    Meant for a launcher's FINAL launch options (robot None: the catalogue default; no preset and no task: the
    backend's default launch, wall/anchor-holes), called before the session folder is made. The backend runs the same
    two checks when it is built, so today an impossible robot/task pair is refused there, after `gar run` has created
    the session folder; nothing in the shared launcher calls this yet (requested: gym_anything_robotics/sessions.py
    prepare)."""
    from drilling.config import TASKS
    from robots.catalog import ENVIRONMENTS
    alias = validate(ENVIRONMENTS["drilling"]["default"] if robot is None else robot)
    options = dict(backend_options or {})
    options.update(options.get("config") or {})
    preset, task = options.get("preset"), options.get("task")
    if preset is None:
        preset = next((p for p, names in TASKS.items() if task in names), None) if task is not None else "wall"
    if preset in TASKS:                                   # an unknown preset or task is the backend's refusal
        check_task(alias, preset, task if task is not None else TASKS[preset][0])
    return alias

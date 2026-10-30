"""Robot-setup validator for gift-bags (robots.catalog ENVIRONMENTS["gift-bags"]["validator"]).

Accepts a catalog alias, a rig id, a gar.robot-setup.v1 object or a RobotSetup when its content (ignoring id,
description and notes) is one of the qualified configurations, and returns that configuration's alias:

  * two FR3 arms, two Panda arms, one FR3 or one Panda on the left mount; each arm with one Franka Hand on its flange
    (mount profile franka_native) and the label_application controller profile, on the counter/table pedestal mounts.

NOT qualified: the Unitree G1 (rig gift-g1: g1@1 with two Dex3-1 hands on the floor mount, barcode_scanning controller
profile). It is built and stands hands-off (gift_bags/g1.py), but with no balance controller it
falls when its arms move toward the work, so it is refused with that measured reason (the catalog's "unsupported"
entry), under any id (scan-g1 is the same hardware). GIFT_BAGS_G1_DEV=1 lets the standing and reach probes build it.

A setup that is the same hardware under another name (the catalog's label-fr3 rig is gift-fr3 with another id) is
accepted. Anything else is refused with the physical reason
(no silent substitution): the paper contact world couples a two-finger parallel gripper along one closing axis with
fingertip pads (gift_bags/world.py, rig.py), the arm IK looks for the Franka hand TCP (kinematics.py), and the pedestal
mounts, hover ready pose and joint-speed reflexes are written for the Franka family.
Another humanoid would need what the G1 has here: a floating-base model whose controller profile stands it on its own
servos (measured), a per-preset standing layout with an arm posture clear of the work surface (scene.G1_LAYOUT), an arm
IK on its hand frames (g1.G1Kinematics), every link meshed as a driven obstacle with its hand material in the friction
ledger (params.FRICTION_DEX3), a hand-posture map for gripper commands (g1.HAND_OPEN / HAND_CLOSED) and a head camera."""

QUALIFIED = {"dual-fr3": "gift-fr3", "dual-panda": "gift-panda", "single-fr3": "gift-fr3-single",
             "single-panda": "gift-panda-single"}
# The G1 is built and stands (gift-g1, g1.py) but is not qualified: with no balance controller it falls as soon as its
# arms move toward the work (catalog "unsupported" gives the measured reason). The standing and reach probes and the
# tests reach it with GIFT_BAGS_G1_DEV=1; nothing else does.
G1_RIG = "gift-g1"


def g1_dev():
    import os
    return os.environ.get("GIFT_BAGS_G1_DEV") == "1"


def resolve_setup(robot):
    """The robot setup for a Config.robot value: the catalog's choice, or the unqualified G1 under GIFT_BAGS_G1_DEV=1."""
    from robots.catalog import for_environment, resolve
    if robot == "g1" and g1_dev():
        return resolve(G1_RIG)
    return for_environment("gift-bags", robot)
MODELS = ("fr3@1", "panda@1")
SINGLE = ("single-fr3", "single-panda")
HUMANOID = ("g1",)
HUMANOID_MODELS = ("g1@1",)


def _content(d):
    return {k: v for k, v in d.items() if k not in ("id", "description", "notes")}


def _humanoid_problems(inst, att):
    """What a humanoid setup lacks to be the qualified G1 configuration (gift-g1)."""
    problems = []
    models = sorted(str(i.get("model")) for i in inst.values())
    if len(inst) != 1 or "g1" not in inst:
        problems.append("exactly one humanoid instance named 'g1' (got " + ", ".join(sorted(inst)) + ")")
    if models != ["g1@1"]:
        problems.append(f"the model g1@1 (got {models}): only the Unitree G1 with Dex3-1 hands has a standing layout, "
                        "arm IK, hand postures and a hand friction class here")
    for name, i in inst.items():
        if i.get("mount") != "g1_floor_mount":
            problems.append(f"{name} on the floor mount g1_floor_mount (got {i.get('mount')!r}): the robot stands on the "
                            "floor at the per-preset layout scene.G1_LAYOUT")
        if i.get("controller_profile") != "barcode_scanning":
            problems.append(f"{name}: controller profile 'barcode_scanning' (got {i.get('controller_profile')!r}): the "
                            "standing gains measured to stand the floating G1 here; another "
                            "profile would need its own standing measurement")
    if att:
        problems.append("nothing attached (the Dex3-1 hands are part of the g1@1 model)")
    return problems


def validate(selected):
    """Return the qualified alias for `selected` or raise ValueError with what the setup would need."""
    from robots.catalog import ENVIRONMENTS, resolve
    item = ENVIRONMENTS["gift-bags"]
    is_g1 = selected in ("g1", G1_RIG)
    if not is_g1 and not (isinstance(selected, str) and selected in item.get("unsupported", {})):
        try:
            is_g1 = _content(resolve(selected).to_dict()) == _content(resolve(G1_RIG).to_dict())
        except (TypeError, ValueError, KeyError, FileNotFoundError):
            is_g1 = False
    if is_g1:
        if g1_dev():
            return "g1"
        raise ValueError(f"gift-bags does not support robot {selected!r}: {item['unsupported']['g1']}")
    if isinstance(selected, str):
        if selected in item.get("unsupported", {}):
            raise ValueError(f"gift-bags does not support robot {selected!r}: {item['unsupported'][selected]}")
        if selected in QUALIFIED:
            return selected
        for alias, rig in QUALIFIED.items():
            if selected == rig:
                return alias
    catalog_error = None
    try:
        d = resolve(selected).to_dict()
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
        # a hand-built setup the catalog cannot resolve (e.g. another gripper with a mount profile it does not list)
        # still gets this environment's own, physical reasons below; the catalog's message is added to them
        if not (isinstance(selected, dict) and all(isinstance(selected.get(k, {}), dict) for k in ("instances", "attachments"))
                and all(isinstance(v, dict) for k in ("instances", "attachments") for v in selected.get(k, {}).values())):
            raise ValueError(f"Invalid robot setup for gift-bags: {exc}") from exc
        d, catalog_error = selected, str(exc)
    if catalog_error is None:
        for alias, rig in QUALIFIED.items():
            if _content(resolve(rig).to_dict()) == _content(d):
                return alias
    inst, att = d.get("instances", {}), d.get("attachments", {})
    from robots.catalog import model

    def _floating(i):
        try:
            return model(i["model"]).to_dict().get("root", {}).get("mounting") == "floating"
        except (TypeError, ValueError, KeyError, FileNotFoundError):
            return False
    floating = [i.get("model") for i in inst.values() if _floating(i)]
    if floating:
        # a floating base is not always a humanoid (the Mobile FR3 rigs float too): say what it is
        kind = "humanoid" if any(str(m).startswith("g1") for m in floating) else "floating-base"
        raise ValueError(f"gift-bags cannot host this {kind} setup ({', '.join(map(str, floating))}): every qualified "
                         "configuration stands on fixed pedestals, and the only floating-base robot with a layout here "
                         "is the Unitree G1, for which it would need: "
                         + "; ".join(_humanoid_problems(inst, att) or ["nothing more, but it is not a qualified "
                                                                         "configuration: " + ", ".join(QUALIFIED)]))
    problems = []
    if not 1 <= len(inst) <= 2 or set(inst) - {"left", "right"}:
        problems.append("instances named 'left' (and optionally 'right')")
    if len(inst) == 1 and "left" not in inst:
        problems.append("a single arm on the left mount")
    models = {i.get("model") for i in inst.values()}
    if models - set(MODELS) or len(models) > 1:
        problems.append(f"arms of one model from {MODELS} (got {sorted(map(str, models))}): mounts, ready pose and "
                        "reflexes are qualified for the Franka family only")
    for name, i in inst.items():
        if i.get("mount") != f"{name}_arm_mount":
            problems.append(f"{name} arm on mount {name}_arm_mount (got {i.get('mount')!r})")
        if i.get("controller_profile") != "label_application":
            problems.append(f"{name}: controller profile 'label_application' (the force-mode Franka Hand paper profile)")
        hands = [a for a in att.values() if a.get("parent") == f"{name}/flange"]
        if len(hands) != 1 or hands[0].get("model") != "franka-hand@1" or hands[0].get("mount_profile") != "franka_native":
            problems.append(f"{name}: one franka-hand@1 on {name}/flange, mount profile franka_native (the paper contact "
                            "world couples a two-finger parallel gripper with fingertip pads along one closing axis)")
    if len(att) != len(inst):
        problems.append("exactly one hand per arm and nothing else attached")
    if catalog_error:
        problems.append(f"a setup the robot catalog can resolve (it says: {catalog_error})")
    raise ValueError("gift-bags cannot host this robot setup: " + ("; ".join(problems) or
                     "it matches none of the qualified configurations " + ", ".join(QUALIFIED)))

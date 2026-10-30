"""Robot-setup validator for wine-pouring (robots.catalog ENVIRONMENTS["wine-pouring"]["validator"]).

Accepts a catalog alias, a gar.robot-setup.v1 object or a RobotSetup when it describes one of the qualified
configurations (two FR3 arms, one FR3 arm on the left mount, two Panda arms, one Panda arm on the left mount; each
with a Franka Hand on its flange,
the barcode_scanning controller profile, mounts on the bar pedestal), and returns that configuration's alias. Anything
else is refused with the physical reason (unified contract: no silent substitution). The scene is written for the
Franka family: the pedestal mounts, the ready pose, the joint-speed/torque reflexes of libfranka and the Franka Hand's
finger pads and force-mode grasp are all specific to it."""

QUALIFIED = {"dual-fr3": "pour-fr3", "single-fr3": "pour-fr3-single", "dual-panda": "pour-panda",
             "single-panda": "pour-panda-single"}
MODELS = ("fr3@1", "panda@1")
MOUNTS = ("left_arm_mount", "right_arm_mount")


def _content(d):
    return {k: v for k, v in d.items() if k not in ("id", "description", "notes")}


def validate(selected):
    """Return the qualified alias for `selected` or raise ValueError with what the setup would need."""
    from robots.catalog import ENVIRONMENTS, resolve
    item = ENVIRONMENTS["wine-pouring"]
    if isinstance(selected, str):
        if selected in item.get("unsupported", {}):
            raise ValueError(f"wine-pouring does not support robot {selected!r}: {item['unsupported'][selected]}")
        if selected in QUALIFIED:
            return selected
        for alias, rig in QUALIFIED.items():
            if selected == rig:
                return alias
    try:
        setup = resolve(selected)
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
        raise ValueError(f"Invalid robot setup for wine-pouring: {exc}") from exc
    d = setup.to_dict()
    for alias, rig in QUALIFIED.items():
        if _content(resolve(rig).to_dict()) == _content(d):
            return alias
    inst, att = d.get("instances", {}), d.get("attachments", {})
    problems = []
    if not 1 <= len(inst) <= 2 or set(inst) - {"left", "right"}:
        problems.append("instances named 'left' (and optionally 'right'); single-arm service uses the left mount")
    if len(inst) == 1 and "left" not in inst:
        problems.append("the single arm on the left mount (spots and coasters are laid out for it)")
    models = {i.get("model") for i in inst.values()}
    if models - set(MODELS):
        problems.append(f"arms from {MODELS} (got {sorted(map(str, models))}): mounts, ready pose, reflexes and finger "
                        "pads are qualified for the Franka family only")
    elif len(models) > 1:
        problems.append(f"both arms of the same model (got {sorted(map(str, models))}): each qualified rig pairs two "
                        "arms of one model; a mixed FR3 + Panda pair has not been qualified")
    for name, i in inst.items():
        want = f"{name}_arm_mount"
        if i.get("mount") != want:
            problems.append(f"{name} arm on mount {want!r} (got {i.get('mount')!r})")
        if i.get("controller_profile") != "barcode_scanning":
            problems.append(f"{name}: controller profile 'barcode_scanning' (the qualified FR3/Panda servo profile)")
        hands = [a for a in att.values() if a.get("parent") == f"{name}/flange"]
        if len(hands) != 1 or hands[0].get("model") != "franka-hand@1" or hands[0].get("mount_profile") != "franka_native":
            problems.append(f"{name}: one franka-hand@1 on {name}/flange with mount profile franka_native "
                            "(the bottle grasp needs its 80 mm stroke and force-mode grip)")
    if len(att) != len(inst):
        problems.append("exactly one Franka Hand per arm and no other attachments")
    if not problems:
        problems.append("a configuration equal to one of the qualified rigs " + ", ".join(QUALIFIED.values()))
    raise ValueError("wine-pouring qualifies two FR3, one FR3 (left), two Panda or one Panda (left) arms with Franka "
                     "Hands; this setup "
                     f"needs: {'; '.join(problems)}. Aliases: {', '.join(QUALIFIED)}")

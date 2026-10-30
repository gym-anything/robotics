"""Robot-setup validator for percussion (robots.catalog ENVIRONMENTS["percussion"]["validator"]).

Accepts a catalog alias, a gar.robot-setup.v1 object or a RobotSetup when it describes one of the qualified
configurations (two FR3 arms, one FR3 arm on the right mount, two Panda arms; each with a Franka Hand on its flange,
the barcode_scanning controller profile, mounts on the station; or the floating Unitree G1 with Dex3 hands, the G1's
"barcode_scanning" standing profile, at the preset's g1.LAYOUT placement) and returns that configuration's alias. Anything else is
refused with the physical reason (unified contract: no silent substitution). The station is written for the Franka
family: the cart mounts, the ready pose, libfranka's velocity/acceleration/jerk limits and reflexes, the Franka Hand's
pads (which carry the fingertip vibro-contact proxies) and its force-mode grasp are specific to it."""

QUALIFIED = {"dual-fr3": "percussion-fr3", "single-fr3": "percussion-fr3-single", "dual-panda": "percussion-panda",
             "g1": "percussion-g1"}
MODELS = ("fr3@1", "panda@1")


def _content(d):
    return {k: v for k, v in d.items() if k not in ("id", "description", "notes")}


def validate(selected):
    """Return the qualified alias for `selected` or raise ValueError with what the setup would need."""
    from robots.catalog import ENVIRONMENTS, resolve
    item = ENVIRONMENTS["percussion"]
    if isinstance(selected, str):
        if selected in item.get("unsupported", {}):
            raise ValueError(f"percussion does not support robot {selected!r}: {item['unsupported'][selected]}")
        if selected in QUALIFIED:
            return selected
        for alias, rig in QUALIFIED.items():
            if selected == rig:
                return alias
    try:
        setup = resolve(selected)
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
        raise ValueError(f"Invalid robot setup for percussion: {exc}. Qualified aliases: {', '.join(QUALIFIED)} (or a "
                         f"gar.robot-setup.v1 object equal to one of the rigs {', '.join(QUALIFIED.values())})") from exc
    d = setup.to_dict()
    for alias, rig in QUALIFIED.items():
        if _content(resolve(rig).to_dict()) == _content(d):
            return alias
    inst, att = d.get("instances", {}), d.get("attachments", {})
    problems = []
    if not 1 <= len(inst) <= 2 or set(inst) - {"left", "right"}:
        problems.append("instances named 'left' and 'right' (or 'right' alone: the drum stands on the right)")
    if len(inst) == 1 and "right" not in inst:
        problems.append("the single arm on the right mount (the drum and the resonator bars' right side are its reach)")
    models = {i.get("model") for i in inst.values()}
    if models - set(MODELS) or len(models) > 1:
        problems.append(f"arms of one model from {MODELS} (got {sorted(map(str, models))}): mounts, ready pose, rate "
                        "limits, reflexes and finger pads are qualified for the Franka family only")
    for name, i in inst.items():
        if i.get("mount") != f"{name}_arm_mount":
            problems.append(f"{name} arm on mount {name}_arm_mount (got {i.get('mount')!r})")
        if i.get("controller_profile") != "barcode_scanning":
            problems.append(f"{name}: controller profile 'barcode_scanning' (the qualified FR3/Panda servo profile)")
        hands = [a for a in att.values() if a.get("parent") == f"{name}/flange"]
        if len(hands) != 1 or hands[0].get("model") != "franka-hand@1" or hands[0].get("mount_profile") != "franka_native":
            problems.append(f"{name}: one franka-hand@1 on {name}/flange with mount profile franka_native")
    if len(att) != len(inst):
        problems.append("exactly one Franka Hand per arm and no other attachments")
    if not problems:
        # meets every structural requirement but is not one of the four qualified rigs (e.g. a single Panda): only
        # those were built and tested, so only those are accepted (the message says so, rather than reading as if a
        # requirement were missing)
        raise ValueError("percussion accepts only its four qualified rigs ("
                         + ", ".join(f"{a} = {r}" for a, r in QUALIFIED.items()) + "); this setup is not equal to any of "
                         "them (a single Panda arm, for example, is not qualified). Aliases: " + ", ".join(QUALIFIED))
    raise ValueError("percussion accepts only its four qualified rigs - two FR3, one FR3 (right) or two Panda arms with "
                     "Franka Hands, or the G1 (rig percussion-g1); this setup "
                     f"needs: {'; '.join(problems)}. Aliases: {', '.join(QUALIFIED)}")

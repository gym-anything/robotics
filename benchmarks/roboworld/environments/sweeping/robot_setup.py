"""Robot-setup validator for sweeping (robots.catalog ENVIRONMENTS["sweeping"]["validator"]).

Accepts a catalog alias, a rig name or a gar.robot-setup.v1 object / RobotSetup when it describes one of the two
built configurations (the Mobile FR3 Duo, or the same robot with one arm: one instance "robot" on the floor_start
mount, controller profile "classroom", no attachments), and returns that configuration's alias. Anything else is
refused with what it would need (unified contract: no silent substitution)."""

QUALIFIED = {"mobile-fr3-duo": "sweep-mobile-fr3-duo", "mobile-fr3-single": "sweep-mobile-fr3-single"}
MODELS = {"mobile-fr3-duo@1": "mobile-fr3-duo", "mobile-fr3-single@1": "mobile-fr3-single"}


def _content(d):
    return {k: v for k, v in d.items() if k not in ("id", "description", "notes", "fingerprint")}


def validate(selected):
    """Return the built alias for `selected` or raise ValueError with what the setup would need."""
    from robots.catalog import ENVIRONMENTS, resolve
    item = ENVIRONMENTS["sweeping"]
    if isinstance(selected, str):
        if selected in item.get("unsupported", {}):
            raise ValueError(f"sweeping does not support robot {selected!r}: {item['unsupported'][selected]}")
        if selected in QUALIFIED:
            return selected
        for alias, rig in QUALIFIED.items():
            if selected == rig:
                return alias
    try:
        setup = resolve(selected)
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
        raise ValueError(f"Invalid robot setup for sweeping: {exc}") from exc
    d = setup.to_dict()
    for alias, rig in QUALIFIED.items():
        if _content(resolve(rig).to_dict()) == _content(d):
            return alias
    inst = d.get("instances", {})
    problems = []
    if list(inst) != ["robot"]:
        problems.append("exactly one instance named 'robot'")
    for i in inst.values():
        if i.get("model") not in MODELS:
            problems.append(f"model in {sorted(MODELS)} (got {i.get('model')})")
        if i.get("mount") != "floor_start":
            problems.append("mount 'floor_start'")
        if i.get("controller_profile") != "classroom":
            problems.append("controller profile 'classroom' (the only one built here)")
    if d.get("attachments"):
        problems.append("no attachments (the Franka Hands are part of the robot description)")
    if not problems:
        problems.append("a configuration equal to one of the built rigs " + ", ".join(QUALIFIED.values()))
    raise ValueError("sweeping builds the Mobile FR3 Duo and its single-arm configuration only; this setup needs: "
                     f"{'; '.join(problems)}. Aliases: {', '.join(QUALIFIED)}")

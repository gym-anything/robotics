"""Robot-setup validator for painting (robots.catalog ENVIRONMENTS["painting"]["validator"]).

Accepts a catalog alias, a rig name or a gar.robot-setup.v1 object / RobotSetup describing one of the qualified
configurations: the Franka Mobile FR3 Duo, or the same platform with one (right) arm, as one instance "robot" on the
floor_start mount with the classroom controller profile (classroom_arrangement's qualified gains, reflexes and sensors)
or the pinned G1 with Dex3 hands for craft tasks. Returns the alias; anything else is refused with what it would need (unified contract: no
silent substitution).
"""

QUALIFIED = {"mobile-fr3-duo": "painting-mobile-fr3-duo", "mobile-fr3-single": "painting-mobile-fr3-single", "g1": "painting-g1"}
MODELS = {"mobile-fr3-duo@1": "mobile-fr3-duo", "mobile-fr3-single@1": "mobile-fr3-single"}


def _content(d):
    return {k: v for k, v in d.items() if k not in ("id", "description", "notes", "fingerprint")}


def validate(selected):
    from robots.catalog import ENVIRONMENTS, resolve
    item = ENVIRONMENTS["painting"]
    if isinstance(selected, str):
        if selected in item.get("unsupported", {}):
            raise ValueError(f"painting does not support robot {selected!r}: {item['unsupported'][selected]}")
        if selected in QUALIFIED:
            return selected
        for alias, rig in QUALIFIED.items():
            if selected == rig:
                return alias
    try:
        setup = resolve(selected)
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
        raise ValueError(f"Invalid robot setup for painting: {exc}") from exc
    d = setup.to_dict()
    for alias, rig in QUALIFIED.items():
        if _content(resolve(rig).to_dict()) == _content(d):
            return alias
    raise ValueError("painting requires a configuration equal to a qualified rig: " + ", ".join(QUALIFIED.values()))


def check_task(robot, preset):
    """Refuse a qualified robot for a preset it cannot do as instructed (catalog ENVIRONMENTS['painting']['task_limits']):
    the single-arm configuration cannot hold an aerosol can and press its button."""
    from robots.catalog import ENVIRONMENTS
    name = validate(robot)
    reason = ENVIRONMENTS["painting"].get("task_limits", {}).get(name, {}).get(preset)
    if reason:
        raise ValueError(f"painting does not support {name!r} on the {preset} preset: {reason}")

"""Robot-setup validator for surface-rinsing (robots.catalog ENVIRONMENTS["surface-rinsing"]["validator"]).

Accepts a catalog alias, a rig name or a gar.robot-setup.v1 object / RobotSetup describing one of the qualified
configurations: the Franka Mobile FR3 Duo (default) or the same platform with one (right) arm, as one instance "robot" on
the floor_start mount with the classroom controller profile (classroom_arrangement's qualified gains, reflexes and
sensors). Returns the alias; anything else is refused with what it would need (unified contract: no silent substitution).
"""

QUALIFIED = {"mobile-fr3-duo": "rinsing-mobile-fr3-duo", "mobile-fr3-single": "rinsing-mobile-fr3-single"}


def _content(d):
    return {k: v for k, v in d.items() if k not in ("id", "description", "notes", "fingerprint")}


def validate(selected):
    from robots.catalog import ENVIRONMENTS, resolve
    item = ENVIRONMENTS["surface-rinsing"]
    if isinstance(selected, str):
        if selected in item.get("unsupported", {}):
            raise ValueError(f"surface-rinsing does not support robot {selected!r}: {item['unsupported'][selected]}")
        if selected in QUALIFIED:
            return selected
        for alias, rig in QUALIFIED.items():
            if selected == rig:
                return alias
    try:
        setup = resolve(selected)
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
        raise ValueError(f"Invalid robot setup for surface-rinsing: {exc}") from exc
    d = setup.to_dict()
    for alias, rig in QUALIFIED.items():
        if _content(resolve(rig).to_dict()) == _content(d):
            return alias
    raise ValueError("surface-rinsing requires a configuration equal to a qualified rig: " + ", ".join(QUALIFIED.values()))


def check_task(robot, task):
    """Refuse a qualified robot for a task it cannot do as instructed (catalog task_limits)."""
    from robots.catalog import ENVIRONMENTS
    name = validate(robot)
    reason = ENVIRONMENTS["surface-rinsing"].get("task_limits", {}).get(name, {}).get(task)
    if reason:
        raise ValueError(f"surface-rinsing does not support {name!r} on {task!r}: {reason}")

"""Robot-setup validator for soil-excavation (robots.catalog ENVIRONMENTS["soil-excavation"]["validator"]).

Accepts a catalog alias, a rig name or a gar.robot-setup.v1 object / RobotSetup describing one of the two qualified
configurations: the Franka Mobile FR3 Duo, or the same platform with one (right) arm, as one instance "robot" on the
floor_start mount with the classroom controller profile (classroom_arrangement's qualified gains, reflexes and sensors)
and no attachments. Returns the alias; anything else is refused with what it would need (unified contract: no silent
substitution). check_task refuses a qualified robot for a task it cannot do as instructed (two-handed tools).
"""

QUALIFIED = {"mobile-fr3-duo": "soil-mobile-fr3-duo", "mobile-fr3-single": "soil-mobile-fr3-single"}
MODELS = {"mobile-fr3-duo@1": "mobile-fr3-duo", "mobile-fr3-single@1": "mobile-fr3-single"}


def _content(d):
    return {k: v for k, v in d.items() if k not in ("id", "description", "notes", "fingerprint")}


def validate(selected):
    from robots.catalog import ENVIRONMENTS, resolve
    item = ENVIRONMENTS["soil-excavation"]
    if isinstance(selected, str):
        if selected in item.get("unsupported", {}):
            raise ValueError(f"soil-excavation does not support robot {selected!r}: {item['unsupported'][selected]}")
        if selected in QUALIFIED:
            return selected
        for alias, rig in QUALIFIED.items():
            if selected == rig:
                return alias
    try:
        setup = resolve(selected)
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
        raise ValueError(f"Invalid robot setup for soil-excavation: {exc}") from exc
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
            problems.append("controller profile 'classroom' (the only qualified one)")
    if d.get("attachments"):
        problems.append("no attachments (the Franka Hands are part of the Duo description)")
    if not problems:
        problems.append("a configuration equal to one of the qualified rigs " + ", ".join(QUALIFIED.values()))
    raise ValueError("soil-excavation qualifies the Mobile FR3 Duo and its single-arm configuration only; this setup "
                     f"needs: {'; '.join(problems)}. Aliases: {', '.join(QUALIFIED)}")


def robot_name(robot):
    if isinstance(robot, str):
        return robot
    if isinstance(robot, dict):
        models = [i.get("model") for i in robot.get("instances", {}).values()]
        return "mobile-fr3-single" if models == ["mobile-fr3-single@1"] else "mobile-fr3-duo"
    return "mobile-fr3-duo"


def check_task(robot, task):
    """Two-handed tools (spade, trenching shovel, post-hole digger, drill with side handle) and the two-tool backfill
    tasks need both hands; the single-arm configuration is offered the one-handed tasks only."""
    from .layout import TASKS
    name = robot_name(robot)
    allowed = TASKS[task]["robots"]
    if name not in allowed:
        held = TASKS[task]["held"]
        why = ("it holds the " + held["both"].replace("_", " ") + " with two hands") if "both" in held else \
              ("it holds " + " and ".join(held.values()) + " at once")
        raise ValueError(f"soil-excavation task {task!r} needs {', '.join(allowed)}: {why}")

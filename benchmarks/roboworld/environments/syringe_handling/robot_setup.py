"""Robot-setup validator for syringe-handling (robots.catalog ENVIRONMENTS["syringe-handling"]["validator"]).

Accepts a catalogue alias, a gar.robot-setup.v1 object or a RobotSetup when it describes an offered configuration and
returns that configuration's alias; anything else is refused with what it would need (unified contract: no silent
substitution). The offered configurations are fixed-base arms with a Franka Hand on the docked cart's two mounts:

  dual-fr3                 two arms (left_arm_mount, right_arm_mount): every task
  dual-panda               two Pandas on the same mounts: sharps-away only (PANDA_TASKS), the one task they have passed
  single-fr3               the right arm alone: sharps-away only (SINGLE_TASKS)

A syringe is held by its barrel in one hand while the other works its plunger, its sheath, its cap or the fitting it
goes on, and nothing in the bay holds either for the robot; so one arm is offered for the one task that is carrying.
"""

QUALIFIED = {"dual-fr3": "syringe-fr3", "dual-panda": "syringe-panda", "single-fr3": "syringe-fr3-single", "g1": "syringe-g1"}
MODELS = ("fr3@1", "panda@1")
HAND = "franka-hand@1"
PROFILE = {"fr3@1": "barcode_scanning", "panda@1": "drilling"}          # arm model -> controller profile
ARMS = {"dual-fr3": ("left", "right"), "dual-panda": ("left", "right"), "single-fr3": ("right",), "g1": ()}
SINGLE_TASKS = ("sharps-away",)
# the two Pandas, driven by the operators written for the FR3 pair, passed sharps-away; set-dose ended with two
# protective stops (85 N between the arms) and 2.80 mL left in the syringe, inject found no clear way for the left arm
# to the pad, and the lock and vial tasks were not tried: they are offered for what passed
PANDA_TASKS = ("sharps-away",)
REPORT = "syringe_handling/REPORT.pdf"


def _content(d):
    return {k: v for k, v in d.items() if k not in ("id", "description", "notes")}


def arms_of(alias):
    return ARMS[alias]


def tasks_for(alias):
    """The tasks an offered alias is offered for (None: every task)."""
    return {"single-fr3": SINGLE_TASKS, "dual-panda": PANDA_TASKS, "g1": SINGLE_TASKS}.get(alias)


def check_task(alias, preset, task=None):
    """Refuse a task a configuration is not offered for, with the reason. (preset, task) as the backend gives them; a
    task alone is accepted too."""
    task = preset if task is None else task
    if alias == "g1" and task not in SINGLE_TASKS:
        raise ValueError("G1 is offered for sharps-away only: supported-pelvis Dex3 grasp/carry components; plunger/fitting work is unqualified")
    if alias == "dual-panda" and task not in PANDA_TASKS:
        raise ValueError(f"syringe-handling does not offer 'dual-panda' for task {task!r}: the two Pandas have passed "
                         f"{', '.join(PANDA_TASKS)} only (set-dose ended with two protective stops and 2.80 mL left; inject found "
                         "no clear way to the pad; the lock and vial tasks were not tried). Use dual-fr3")
    if len(ARMS[alias]) == 1 and task not in SINGLE_TASKS:
        raise ValueError(f"syringe-handling does not offer {alias!r} for task {task!r}: one hand holds the syringe's barrel "
                         "while the other works its plunger, sheath, cap or fitting, and nothing in the bay holds either "
                         f"for the robot. One arm is offered for {', '.join(SINGLE_TASKS)}; use dual-fr3")


def validate(selected):
    """Return the offered alias for `selected` or raise ValueError with what the setup would need."""
    from robots.catalog import ENVIRONMENTS, resolve
    item = ENVIRONMENTS["syringe-handling"]
    if isinstance(selected, str):
        if selected in item.get("unsupported", {}):
            raise ValueError(f"syringe-handling does not support robot {selected!r}: {item['unsupported'][selected]}")
        if selected in QUALIFIED:
            return selected
        for alias, rig in QUALIFIED.items():
            if selected == rig:
                return alias
    try:
        if _content(resolve(selected).to_dict()) == _content(resolve("syringe-g1").to_dict()):
            return "g1"
    except (TypeError, ValueError, KeyError, FileNotFoundError):
        pass
    raw = selected if isinstance(selected, dict) else (selected.to_dict() if hasattr(selected, "to_dict") else None)
    if isinstance(raw, dict):
        other = sorted({str(a.get("model")) for a in (raw.get("attachments") or {}).values()
                        if isinstance(a, dict) and a.get("model") != HAND})
        if other:
            raise ValueError(f"syringe-handling does not support end effector(s) {', '.join(other)}: the grasps of barrels, "
                             f"sheaths, caps and the connector are built and measured for the Franka Hand ({HAND}) only. "
                             f"Aliases: {', '.join(QUALIFIED)}")
        insts = [i for i in (raw.get("instances") or {}).values() if isinstance(i, dict)]
        arms = sorted({str(i.get("model")) for i in insts} - set(MODELS))
        if arms:
            raise ValueError(f"syringe-handling does not support arm model(s) {', '.join(arms)}: the bay mounts fixed-base "
                             f"{' or '.join(MODELS)} arms on the docked cart ({REPORT}). Aliases: {', '.join(QUALIFIED)}")
    try:
        setup = resolve(selected)
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
        if isinstance(selected, str):
            refused = ", ".join(sorted(item.get("unsupported", {})))
            raise ValueError(f"syringe-handling has no robot setup {selected!r}: it builds the rigs "
                             f"{', '.join(f'{a} ({r})' for a, r in QUALIFIED.items())} or an equal gar.robot-setup.v1 "
                             f"object ({REPORT}); refused with stated reasons: {refused}") from exc
        raise ValueError(f"Invalid robot setup for syringe-handling: {exc}") from exc
    d = setup.to_dict()
    for alias, rig in QUALIFIED.items():
        if _content(resolve(rig).to_dict()) == _content(d):
            return alias
    inst, att = d.get("instances", {}), d.get("attachments", {})
    problems = []
    if not 1 <= len(inst) <= 2 or set(inst) - {"left", "right"}:
        problems.append("instances named 'left' and 'right', or 'right' alone")
    if len({i.get("model") for i in inst.values()}) > 1:
        problems.append("both arms of the same model (a mixed FR3/Panda pair was not built)")
    if set(inst) == {"left"}:
        problems.append("an arm alone on the right mount (single-fr3); no left arm alone is offered")
    for name, i in inst.items():
        if i.get("model") not in MODELS:
            problems.append(f"{name}: arm model from {MODELS}")
        if i.get("mount") != f"{name}_arm_mount":
            problems.append(f"{name} arm on mount {name}_arm_mount (got {i.get('mount')!r})")
        want = PROFILE.get(i.get("model"))
        if want is not None and i.get("controller_profile") != want:
            problems.append(f"{name}: controller profile {want!r} (got {i.get('controller_profile')!r})")
        if len([a for a in att.values() if a.get("parent") == f"{name}/flange"]) != 1:
            problems.append(f"{name}: one {HAND} on {name}/flange")
    if set(inst) == {"right"} and next(iter(inst.values())).get("model") == "panda@1":
        problems.append("a single Panda was not built (single-fr3 is the one-arm rig)")
    raise ValueError("syringe-handling does not offer this robot setup; it needs: " + "; ".join(problems or ["one of the rigs "
                     + ", ".join(QUALIFIED.values())]) + f". Aliases: {', '.join(QUALIFIED)}")

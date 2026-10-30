"""Robot-setup validator for ingredient-mixing (robots.catalog ENVIRONMENTS["ingredient-mixing"]["validator"]).

Accepts a catalog alias, a gar.robot-setup.v1 object or a RobotSetup when it describes an offered configuration and
returns that configuration's alias; anything else is refused with what it would need (unified contract: no silent
substitution). The offered configurations are fixed-base arms with a Franka Hand on the docked cart's two mounts:

  dual-fr3, dual-panda         two arms (left_arm_mount, right_arm_mount)
  single-fr3, single-panda     the right arm alone

The left FR3 alone (rig mix-fr3-single-left, NOT_OFFERED) builds and is offered for no task (LEFT_TASKS): it is not a
choice of the catalogue, and its name, its rig's name and an equal setup object are refused with the reason. Its code
path stays: ARMS, tasks_for and check_task know it, and a probe builds it by its rig's name.

The FR3 rigs take the arm servos and the hand from the FR3's barcode_scanning profile; the Panda rigs take them from
the Panda's `drilling` profile, which is the Panda's barcode_scanning arm joints with the FR3 hand's finger controller
(35 N per finger; the Panda's own barcode_scanning profile limits the hand to 20 N per finger, which this
environment's grasps exceed: a scripted dry-whisk rollout stops at its first grasp with it). Which tasks an arm alone is
offered for follows where things stand (check_task): the crock stands to the right of the vessel and the mixer to its
left, and an arm alone cannot hold the pan's handle while it stirs.

What stands behind each offer: every rig with a scripted rollout (the motions without the mixture), the inverse
kinematics of each arm to the other arm's side, and the left hand alone on the standing mixer's switch."""

QUALIFIED = {"dual-fr3": "mix-fr3", "single-fr3": "mix-fr3-single",
             "dual-panda": "mix-panda", "single-panda": "mix-panda-single"}
# Built and not offered: name -> rig. Not a catalogue alias; refused at launch with the catalogue's stated reason
# (robots.catalog ENVIRONMENTS["ingredient-mixing"]["unsupported"]). To offer it: give LEFT_TASKS a task that a rollout
# stands behind, move the entry to QUALIFIED and to the catalogue's aliases, and take its `unsupported` text out.
NOT_OFFERED = {"single-fr3-left": "mix-fr3-single-left"}
MODELS = ("fr3@1", "panda@1")
HAND = "franka-hand@1"
PROFILE = {"fr3@1": "barcode_scanning", "panda@1": "drilling"}          # arm model -> controller profile
ARMS = {"dual-fr3": ("left", "right"), "dual-panda": ("left", "right"), "single-fr3": ("right",),
        "single-panda": ("right",), "single-fr3-left": ("left",)}
REPORT = "ingredient_mixing/robot_setup.py: robot configurations"
# One arm on the right mount: the tasks worked with a tool from the crock (which stands to the vessel's right) in a
# bowl that nobody holds. Scripted rollouts did dry-whisk, sauce-blend, muffin-fold and pancake-whisk this way
# (the left arm parked; the bowl moved 0.02 to 11.5 mm against a limit of 50 mm); pudding-whisk is the same bowl,
# whisk, crock and start layout as pancake-whisk (scene.TASK_SETUP) and is offered on that ground, without a rollout
# of its own by any rig.
RIGHT_TASKS = ("dry-whisk", "sauce-blend", "muffin-fold", "pancake-whisk", "pudding-whisk")
# One arm on the left mount: offered for no task. The hand mixer stands on its heel to the vessel's left and the left
# hand takes and carries it, but the task as the maker describes it needs a second hand: the mixer is held in the
# batter while the other hand slides its switch on, up and off. Alone, the left hand can slide
# the switch only while the mixer stands on the counter (5.0 mm of slide with
# 10 to 12 N on the fingers, beaters at 260 rpm, the mixer moved 0.01 mm; switching off again cost four protective
# stops of the arm against its own links), which means carrying turning beaters into and out of the batter,
# and standing a running mixer back on its heel; no one-armed rollout of that has been run, and the two-armed rollout
# does not stand the mixer back up either. The crock's tools are out of the left arm's reach. The rig
# builds and is refused with these reasons until a rollout stands behind an offer.
LEFT_TASKS = ()
SINGLE_TASKS = RIGHT_TASKS                                               # the former name (right-arm rigs)
HAND_ONLY = (f"the grasps of handles, rims and the mixer are built and measured for the Franka Hand ({HAND}) only "
             "(pads from that hand's catalogue data, ingredient_mixing/scene.py)")
OFFERED = ("two arms (instances 'left' and 'right') or the right one alone, each an FR3 (controller profile "
           f"{PROFILE['fr3@1']!r}) or a Panda (controller profile {PROFILE['panda@1']!r}) of the same model, on "
           "left_arm_mount / right_arm_mount, each with one Franka Hand on its flange")


def _content(d):
    return {k: v for k, v in d.items() if k not in ("id", "description", "notes")}


def arms_of(alias):
    """The arm instances an offered alias builds (('left', 'right'), ('right',) or ('left',))."""
    return ARMS[alias]


def _not_offered(name, what):
    from robots.catalog import ENVIRONMENTS
    reason = ENVIRONMENTS["ingredient-mixing"].get("unsupported", {}).get(name, "it is offered for no task")
    return ValueError(f"ingredient-mixing does not support robot {what}: {reason}")


def validate(selected):
    """Return the offered alias for `selected` or raise ValueError with what the setup would need."""
    from robots.catalog import ENVIRONMENTS, resolve
    item = ENVIRONMENTS["ingredient-mixing"]
    if isinstance(selected, str):
        if selected in item.get("unsupported", {}):
            raise ValueError(f"ingredient-mixing does not support robot {selected!r}: {item['unsupported'][selected]}")
        if selected in QUALIFIED:
            return selected
        for alias, rig in QUALIFIED.items():
            if selected == rig:
                return alias
        for name, rig in NOT_OFFERED.items():                 # built, offered for no task: refused with its reason
            if selected in (name, rig):
                raise _not_offered(name, f"{selected!r}" + ("" if selected == name else f" ({name})"))
    raw = selected if isinstance(selected, dict) else (selected.to_dict() if hasattr(selected, "to_dict") else None)
    if isinstance(raw, dict):
        other = sorted({str(a.get("model")) for a in (raw.get("attachments") or {}).values()
                        if isinstance(a, dict) and a.get("model") != HAND})
        if other:
            raise ValueError(f"ingredient-mixing does not support end effector(s) {', '.join(other)}: {HAND_ONLY}. "
                             f"Aliases: {', '.join(QUALIFIED)}")
        insts = [i for i in (raw.get("instances") or {}).values() if isinstance(i, dict)]
        arms = sorted({str(i.get("model")) for i in insts} - set(MODELS))
        if arms:
            raise ValueError(f"ingredient-mixing does not support arm model(s) {', '.join(arms)}: the station mounts "
                             f"fixed-base {' or '.join(MODELS)} arms on the docked cart ({REPORT}). "
                             f"Aliases: {', '.join(QUALIFIED)}")
    try:
        setup = resolve(selected)
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
        if isinstance(selected, str):
            refused = ", ".join(sorted(item.get("unsupported", {})))
            raise ValueError(f"ingredient-mixing has no robot setup {selected!r}: it builds the rigs "
                             f"{', '.join(f'{a} ({r})' for a, r in QUALIFIED.items())} or an equal gar.robot-setup.v1 "
                             f"object ({REPORT}); refused with stated reasons: {refused}") from exc
        raise ValueError(f"Invalid robot setup for ingredient-mixing: {exc}") from exc
    d = setup.to_dict()
    for alias, rig in QUALIFIED.items():
        if _content(resolve(rig).to_dict()) == _content(d):
            return alias
    for name, rig in NOT_OFFERED.items():                     # a setup object equal to a rig that is not offered
        if _content(resolve(rig).to_dict()) == _content(d):
            raise _not_offered(name, f"setup {d.get('id')!r} (the content of {rig}, {name})")
    inst, att = d.get("instances", {}), d.get("attachments", {})
    problems = []
    if not 1 <= len(inst) <= 2 or set(inst) - {"left", "right"}:
        problems.append("instances named 'left' and 'right', or 'right' alone")
    models = {i.get("model") for i in inst.values()}
    if len(models) > 1:
        problems.append("both arms of the same model (a mixed FR3/Panda pair was not built)")
    if set(inst) == {"left"}:
        problems.append("an arm alone on the right mount (single-fr3, single-panda); no left arm alone is offered")
    for name, i in inst.items():
        if i.get("model") not in MODELS:
            problems.append(f"{name}: arm model from {MODELS}")
        if i.get("mount") != f"{name}_arm_mount":
            problems.append(f"{name} arm on mount {name}_arm_mount (got {i.get('mount')!r})")
        want = PROFILE.get(i.get("model"))
        if want is not None and i.get("controller_profile") != want:
            problems.append(f"{name}: controller profile {want!r} (got {i.get('controller_profile')!r})")
        hands = [a for a in att.values() if a.get("parent") == f"{name}/flange"]
        if len(hands) != 1 or hands[0].get("model") != HAND or hands[0].get("mount_profile") != "franka_native":
            problems.append(f"{name}: one {HAND} on {name}/flange with mount profile franka_native ({HAND_ONLY})")
    if len(att) != len(inst):
        problems.append("exactly one Franka Hand per arm and no other attachments")
    if not problems:
        problems.append("a configuration equal to one of the rigs " + ", ".join(QUALIFIED.values()))
    raise ValueError(f"ingredient-mixing offers {OFFERED}; this setup needs: {'; '.join(problems)}. "
                     f"Aliases: {', '.join(QUALIFIED)} ({REPORT})")


# The Panda pair is not offered for the hand mixer: rehearsed without the mixture with the scripted two-handed rollout,
# the left hand met the mixer as it reached for
# it (181 N, a protective stop), the two arms met once (41 N), the right hand slid the switch on and up while the left
# held the mixer in the bowl, and could then not reach the switch to turn it off: the rehearsal ended with the mixer
# running. The same rollout goes through with the FR3 pair.
PAIR_NOT_SHOWN = {"dual-panda": ("cake-mixer",)}


def tasks_for(alias):
    """The tasks an alias is offered for (None: every task)."""
    arms = ARMS[alias]
    if len(arms) == 2:
        if alias in PAIR_NOT_SHOWN:
            from .config import TASKS
            return tuple(t for names in TASKS.values() for t in names if t not in PAIR_NOT_SHOWN[alias])
        return None
    return RIGHT_TASKS if arms == ("right",) else LEFT_TASKS


def check_task(alias, preset, task):
    """An arm alone is offered where what it needs stands on its side and nothing has to be held meanwhile."""
    offered = tasks_for(alias)
    if offered is None or task in offered:
        return
    if task in PAIR_NOT_SHOWN.get(alias, ()):
        raise ValueError(f"ingredient-mixing task {preset}/{task} is not offered with {alias}: rehearsed without the "
                         "mixture, the two-handed mixer rollout ended with the mixer running (the right hand could not "
                         "reach the switch to turn it off) after two protective stops "
                         f"({REPORT}). This task is offered "
                         "with dual-fr3")
    two = [a for a in QUALIFIED if len(ARMS[a]) == 2 and task not in PAIR_NOT_SHOWN.get(a, ())]
    others = [a for a in QUALIFIED if len(ARMS[a]) == 1 and a != alias and task in (tasks_for(a) or ())]
    if preset == "hob":
        why = ("on the hob one hand holds the pan's handle while the other stirs, and turns the knob to 0 while the "
               "other stirs on (every hob rollout); a run with the pan unheld has "
               "not been made")
    elif task == "cake-mixer" and ARMS[alias] == ("left",):
        why = ("the mixer is held in the batter by one hand while the other slides its switch on, up and off; the left "
               "hand alone can slide the switch only while the mixer stands on the counter, "
               "turning beaters would then be carried into and out of the "
               "batter, and no one-armed rollout has been run")
    elif task == "cake-mixer":
        why = ("the hand mixer stands 0.34 m to the left of the vessel, 0.79 m from the right arm's base: its handle is "
               "within that arm's reach by inverse kinematics but no rollout has taken "
               "the right arm there, and the mixer is taken and carried by the left hand in every rollout")
    elif ARMS[alias] == ("left",):
        why = ("the utensil crock stands 0.37 m to the right of the vessel, 0.73 to 0.93 m from the left arm's base: of "
               "the three tools standing in it the left hand can take the nearest only ("
               "the farthest handle is 43 to 57 mm out of reach and another cannot be lifted clear of the crock)")
    else:
        why = "this task is not among those offered with the right arm alone"
    raise ValueError(f"ingredient-mixing task {preset}/{task} is not offered with {alias}: {why} ({REPORT}). "
                     f"{alias} is offered for {', '.join(offered) or 'no task'}; this task is offered with "
                     f"{', '.join(two + others)}")

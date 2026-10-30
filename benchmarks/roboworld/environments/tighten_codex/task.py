"""Work instructions and pure task recipes; no physics import at discovery time."""

TASKS={
 "sensor":("Secure the staged W4F replacement sensor",
    "The frame corner is already secured. Check the white reference flag, slide its carriage aside for screwdriver clearance, and hold the loose sensor on its bracket. Use the 2.5 mm hex bit and green 0.35 N m driver to seat both prestarted M3 screws until the driver clicks. Return the flag and recheck detection after release. The W4F mounting maximum is 0.4 N m."),
 "aim":("Aim and lock the photoelectric sensor",
    "The frame corner is already secured. The loose W4F has swung to the end of its bracket's arc slot and its light spot misses the white flag. Roll the sensor about its pivot screw in the arc slot until the spot lands on the flag, and hold it there. Retract the flag for tool access and seat both M3 screws with the 2.5 mm bit and green 0.35 N m driver until it clicks. Return the flag and recheck the spot after release."),
 "frame":("Secure the extrusion corner",
    "The corner is loose: the rail has drooped nose-down onto its upward screw. Lift it level and square and hold it there while you use the 3 mm bit and yellow 0.60 N m driver on the two prestarted M4 fasteners until the driver clicks: one facing forward into the post T-slot and one facing upward into the rail slot. Release it and verify that it stays level and seated."),
 "mixed":("Recommission the loosened sensor stand",
    "The W4F sensor's two M3 screws and the frame corner bracket's two M4 screws (one forward into the post, one upward from underneath into the rail) have worked loose, and the rail has drooped. Tighten the M3s with the green 0.35 N m driver and the M4s with the yellow 0.60 N m driver until each driver clicks, leaving the rail level. The white reference flag slides aside for tool access; when you finish it must be back at its home position, the sensor must detect it there, and the assembly must hold on its own after you let go."),
 "support":("Support and secure a slightly offset rail",
    "The loose rail has drooped and sits 1.5 mm along its slot. Lift it level, return it to the reference position and hold it there, then use the yellow 3 mm/0.60 N m driver on both orthogonal M4 fasteners until it clicks. Release the support and check that it stays level and in place."),
 "access":("Tighten through a restricted service opening",
    "A service guard narrows access to the underside M4 fastener of the loose corner. Support the rail level, seat the 3 mm bit through the opening without striking the guard and tighten with the yellow 0.60 N m driver until it clicks. Verify retention after removing the tool."),
}

LEGACY_TASKS = tuple(TASKS)
EIGHT_CONTRACT = "tighten-tasks-v1"
MIXED_CONTRACT = "tighten-mixed-v1"
LEGACY_CONTRACT = "tighten-legacy-v1"
NEW_TASKS = ("front-m4", "find-loose")
TASKS.update({
 "front-m4": ("Re-seat the forward corner screw",
    "The corner bracket's forward M4 screw into the post T-slot has worked loose. Everything else is tight and the rail is level. Tighten that screw with the yellow 3 mm/0.60 N m driver until it clicks. Leave the other screws tight, the rail level and the reference flag home with the sensor detecting it, after releasing the stand."),
 "find-loose": ("Find and re-seat the loose screw",
    "After a conveyor jam, one of the sensor's two M3 screws or the corner bracket's forward M4 has worked loose. Find it and re-seat it until its preset driver clicks: green 2.5 mm/0.35 N m for M3, yellow 3 mm/0.60 N m for M4. Leave the other screws tight, the rail level and the reference flag home with the sensor detecting it, after releasing the stand."),
})

# Selections without a contract keep the original wording and semantics. These job-level
# instructions apply only to the explicitly selected eight-task contract.
EIGHT_INSTRUCTIONS = {
 "sensor": "Secure the staged W4F replacement sensor with both prestarted M3 screws and the green 2.5 mm/0.35 N m driver, until it clicks on each screw. The frame is already secured. Leave the reference flag at home with the sensor detecting it and holding on its own after release. The flag slides aside for tool access; the W4F mounting maximum is 0.4 N m.",
 "aim": "Aim and lock the photoelectric sensor. The frame is already secured; the loose W4F at its arc-slot end misses the white flag. Secure both M3 screws with the green 2.5 mm/0.35 N m driver until it clicks on each. Leave the sensor detecting the flag at its original home position and holding on its own after release. The flag slides aside for tool access.",
 "frame": "Secure the loose extrusion corner with the yellow 3 mm/0.60 N m driver until it clicks on both prestarted M4 screws: one forward into the post T-slot and one upward into the rail slot. Leave the drooped rail level, square and seated at its reference position, holding on its own after release.",
 "access": "Secure the underside M4 screw through the restricted service opening with the yellow 3 mm/0.60 N m driver until it clicks. Leave the rail level and holding on its own after removing the tool and releasing the stand.",
 "support": "Secure the offset extrusion rail. It has drooped and sits 1.5 mm along its slot. Tighten both orthogonal M4 screws with the yellow 3 mm/0.60 N m driver until it clicks on each. Leave the rail level, square and at its reference position, holding on its own after release.",
}
EIGHT_INSTRUCTIONS.update({n: f"{TASKS[n][0]}. {TASKS[n][1]}" for n in NEW_TASKS})
EIGHT_ORDER = ("front-m4", "sensor", "find-loose", "frame", "access", "support", "aim", "mixed")


def contract_for(task, selected=""):
    """Validate a contract without changing how the other selectors are coerced."""
    if type(selected) is not str:
        raise ValueError("contract must be a string")
    default = MIXED_CONTRACT if task == "mixed" else EIGHT_CONTRACT if task in NEW_TASKS else LEGACY_CONTRACT
    allowed = (default, EIGHT_CONTRACT) if task in LEGACY_TASKS and task != "mixed" else (default,)
    if selected and selected not in allowed:
        raise ValueError(f"Unknown contract {selected!r} for {task}: {allowed}")
    return selected or default


def uses_eight(config):
    return contract_for(config.task, getattr(config, "task_contract", "")) == EIGHT_CONTRACT


def instruction(config):
    if uses_eight(config):
        return EIGHT_INSTRUCTIONS[config.task]
    title,text=TASKS[config.task]
    return f"{title}. {text}"


def resolve(selection):
    if set(selection)-{"task","seed","contract"}:
        raise ValueError("Selectors are task, seed and contract")
    task=selection.get("task","mixed")
    seed=selection.get("seed",0)
    if task not in TASKS:
        raise ValueError(f"Unknown tighten task: {task}")
    if type(seed) is not int or seed<0:
        raise ValueError("seed must be a nonnegative integer")
    chosen=contract_for(task,selection.get("contract",""))
    explicit=chosen==EIGHT_CONTRACT and task in LEGACY_TASKS
    config={"task":task,"seed":seed}
    picked=dict(config)
    if explicit:
        config["task_contract"]=chosen
        picked["contract"]=chosen
    from types import SimpleNamespace
    title,text=TASKS[task]
    suffix=f"/{chosen}" if explicit else ""
    return dict(task_id=f"tighten-codex/v1/{task}{suffix}/seed-{seed}",
        description=instruction(SimpleNamespace(**config)) if chosen==EIGHT_CONTRACT else f"{title}: {text}",
        backend_options={"config":config}, selection=picked,split=None)

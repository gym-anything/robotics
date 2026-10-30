"""The opt-in eight-task contract; no placement or physics changes.

Bare selections retain the original eleven tasks, instructions, scores and IDs.
behind-radiator keeps its own progress-v1 contract and is outside the new one.
"""

CONTRACT = "painting-tasks-v1"
LEGACY = "checks-v1"
RADIATOR = "progress-v1"            # task.BEHIND_RADIATOR["contract"]
EIGHT_TASKS = ("load-roller", "swatch-card", "roll-strip", "clear-coat", "cut-in-trim",
               "spray-stencil", "brush-stroke", "behind-radiator")
DIFFICULTY = dict(zip(EIGHT_TASKS, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))
INSTRUCTIONS = {
    "load-roller": "Load the dry roller cover evenly with paint from the tray, then hold it above the tray, ready to roll, without dripping.",
    "swatch-card": "Paint the pencilled 50 mm square on the watercolour paper solid with the blue acrylic from the palette.",
    "roll-strip": "Roll the strip of wall between the two blue tape marks, from the brushed band under the ceiling down to the band above the baseboard, to full hide, joining the wet strip on its left and keeping paint off the ceiling, trim and outlet plate; leave the roller resting in the tray.",
    "clear-coat": "Clear-coat the photo print lying in the spray box evenly with the aerosol lacquer, keeping overspray inside the box.",
    "cut-in-trim": "With the angled sash brush, cut in along the left side of the window casing and around the outlet plate, a band at least 40 mm wide, without paint on the casing or the plate.",
    "spray-stencil": "Spray the number 3 through the stencil onto the crate side in the spray box with an even coat and sharp edges, keeping overspray inside the box.",
    "brush-stroke": "With the round brush, paint the pencilled S-stroke on the paper in the red acrylic, swelling from thin to wide and back as the pencilled outline does, and staying inside it.",
}


def selected_contract(task, contract=None):
    default = RADIATOR if task == "behind-radiator" else LEGACY
    if contract is None or contract == default:
        return default
    if contract == CONTRACT and task in INSTRUCTIONS:
        return CONTRACT
    raise ValueError(f"Unknown contract {contract!r} for {task}; choose {default}"
                     + (f" or {CONTRACT}" if task in INSTRUCTIONS else ""))


def describe(task, seed=0, contract=None):
    from .layout import TASKS, DEEP_COLORS, PAINT_COLORS
    if selected_contract(task, contract) != CONTRACT:
        return TASKS[task]["instruction"]
    text = INSTRUCTIONS[task]
    if TASKS[task]["preset"] == "wall" and task != "load-roller" and PAINT_COLORS[seed % 3] in DEEP_COLORS:
        text += " The wall already has its first coat of this colour (dry); this is the second, finish coat."
    return text


def apply_contract(spec, contract=None):
    """Change task metadata only, including for caller-supplied diagnostic specs."""
    chosen = selected_contract(spec["task"], contract)
    if chosen == CONTRACT:
        spec = dict(spec, instruction=describe(spec["task"], spec["seed"], chosen))
    return spec


def selections():
    from .layout import TASKS
    return [dict(preset=TASKS[t]["preset"], task=t, seed=0,
                 **({"contract": CONTRACT} if t in INSTRUCTIONS else {})) for t in EIGHT_TASKS]

"""Start of recap-and-tidy: a clean board, the six markers in the tray (presets.tool_poses) with the caps of three of
them (OPEN_COLORS) pulled off along the barrel CAP_START_OFFSET_M beyond the seated position, the other three capped,
and the wiper in the tray.

setup(task, start=False) gives the task's strokes (task_set/catalogue.py's CatalogueTask.strokes): none, the board is
clean at the start and at the end. cap_offset(color) is how far beyond the seated position a marker's cap starts
(presets.tool_poses, metres). separate_caps(env) is called when the start is applied to a built environment
(presets.apply_start): the open caps lose their former contact state.
"""
OPEN_COLORS = ("black", "red", "blue")
CAP_START_OFFSET_M = .075


def setup(task, start=False):
    return ()


def cap_offset(color):
    return CAP_START_OFFSET_M if color in OPEN_COLORS else 0


def separate_caps(env):
    # Rigid translations leave seated caps' constitutive state unchanged.
    # Only separated caps lose their former contact warm start/deformation.
    for color in OPEN_COLORS:
        env.deformations.pop(f"cap_{color}", None)

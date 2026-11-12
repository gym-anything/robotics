"""Start of correct-misspelling: SCIENSE written in black across the board as dry marker; the six markers capped and the
wiper in the tray (presets.tool_poses). The reference picture shows SCIENCE in its place.

setup(task, start=False) gives the task's strokes (task_set/catalogue.py's CatalogueTask.strokes), each (element,
colour, line weight, points in world metres x, z). With start: the writing on the board at the start, which
presets.seed_board lays down as dry pigment when the episode is built. Without: the desired final board, which the
reference picture shows and the verifier reads.
"""
from whiteboard.task_set.catalogue import text_strokes


def setup(task, start=False):
    return text_strokes("SCIENSE" if start else "SCIENCE", task.center_x, task.center_z, .70)

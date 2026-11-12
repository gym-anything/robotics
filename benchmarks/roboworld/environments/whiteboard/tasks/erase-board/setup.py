"""Start of erase-board: HELLO in black above 12345 in blue, written across the board as dry marker; the six markers
capped and the wiper in the tray (presets.tool_poses).

setup(task, start=False) gives the task's strokes (task_set/catalogue.py's CatalogueTask.strokes), each (element,
colour, line weight, points in world metres x, z). With start: the writing on the board at the start, which
presets.seed_board lays down as dry pigment when the episode is built. Without: the desired final board, here clean,
which the reference picture shows and the verifier reads.
"""
from whiteboard.task_set.catalogue import text_strokes


def setup(task, start=False):
    return (text_strokes("HELLO", -.05, 1.38, .60) +
            text_strokes("12345", -.05, 1.15, .60, "blue")) if start else ()

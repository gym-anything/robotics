"""Start of write-text: a clean board, the six markers capped and the wiper in the tray (presets.tool_poses); the
reference picture shows HELLO in black, 0.60 m wide, at the task's placement (center_x, center_z).

setup(task, start=False) gives the task's strokes (task_set/catalogue.py's CatalogueTask.strokes), each (element,
colour, line weight, points in world metres x, z). With start: the writing on the board at the start, here none
(presets.seed_board lays it down as dry pigment when the episode is built). Without: the desired final board, which the
reference picture shows and the verifier reads.
"""
from whiteboard.task_set.catalogue import text_strokes


def setup(task, start=False):
    return () if start else text_strokes("HELLO", task.center_x, task.center_z, .60)

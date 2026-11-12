"""Start of erase-except-boxed: a blue box holding HW, A+B and 123 in black, and beside it WORK in black above 1234 in
red, all as dry marker; the six markers capped and the wiper in the tray (presets.tool_poses).

setup(task, start=False) gives the task's strokes (task_set/catalogue.py's CatalogueTask.strokes), each (element,
colour, line weight, points in world metres x, z). With start: the writing on the board at the start, which
presets.seed_board lays down as dry pigment when the episode is built. Without: the desired final board, the box and its
contents only, which the reference picture shows and the verifier reads.
"""
from whiteboard.task_set.catalogue import text_strokes


def setup(task, start=False):
    box = (("box", "blue", "normal", ((-.40, 1.12), (-.10, 1.12), (-.10, 1.40), (-.40, 1.40), (-.40, 1.12))),)
    kept = box + tuple(s for word, z in (("HW", 1.34), ("A+B", 1.26), ("123", 1.18))
                       for s in text_strokes(word, -.25, z, .20))
    outside = text_strokes("WORK", .18, 1.34, .38) + text_strokes("1234", .18, 1.18, .38, "red")
    return kept + outside if start else kept

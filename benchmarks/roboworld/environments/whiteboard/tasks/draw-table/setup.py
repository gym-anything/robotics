"""Start of draw-table: a clean board, the six markers capped and the wiper in the tray (presets.tool_poses); the
reference picture shows the blank table to draw in blue, 0.60 m wide (TablePlacement): outer frame, header line and two
column lines.

setup(task, start=False) gives the task's strokes (task_set/catalogue.py's CatalogueTask.strokes), each (element,
colour, line weight, points in world metres x, z). With start: the writing on the board at the start, here none
(presets.seed_board lays it down as dry pigment when the episode is built). Without: the desired final board, which the
reference picture shows and the verifier reads. table_strokes() is the table in units of its width; world_table()
places it on the board.
"""
from dataclasses import dataclass


def table_strokes():
    """Same tuple format as bridge_strokes; metres divided by a 0.60 m width."""
    return tuple((name, "blue", weight, points) for name, weight, points in (
        ("top", "thick", ((-.5, .3), (.5, .3))),
        ("bottom", "thick", ((-.5, -.3), (.5, -.3))),
        ("left", "thick", ((-.5, -.3), (-.5, .3))),
        ("right", "thick", ((.5, -.3), (.5, .3))),
        ("header", "normal", ((-.5, .2), (.5, .2))),
        ("column one", "normal", ((-1/6, -.3), (-1/6, .3))),
        ("column two", "normal", ((1/6, -.3), (1/6, .3))),
    ))


def world_table(task):
    return tuple((n, c, w, tuple((task.center_x + x * task.width_m, task.center_z + z * task.width_m)
                               for x, z in p)) for n, c, w, p in table_strokes())


@dataclass(frozen=True)
class TablePlacement:
    center_x: float = -.05
    center_z: float = 1.28
    width_m: float = .60


def setup(task, start=False):
    return () if start else world_table(TablePlacement())

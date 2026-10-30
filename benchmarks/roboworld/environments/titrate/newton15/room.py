"""Fixed laboratory architecture, shared by collision and rendering.

Room dimensions are declared architectural design choices, not a particular
manufacturer's laboratory CAD. Apparatus and the robot remain separate modules.
"""


def build_room(lab):
    lab.box("floor", (0., -.2, -.10), (4., 3.8, .20), "floor")
    lab.box("ceiling", (0., -.2, 2.85), (4., 3.8, .10), "wall")
    lab.box("side_wall", (-1.9, -.2, 1.4), (.15, 3.6, 2.8), "wall")
    lab.box("right_wall", (1.9, -.2, 1.4), (.15, 3.6, 2.8), "wall")
    # Actual window opening; no opaque wall hidden behind a glass decal.
    for name, position, size in (
        ("rear_left", (-.725, 1.6, 1.4), (2.45, .15, 2.8)),
        ("rear_right", (1.725, 1.6, 1.4), (.45, .15, 2.8)),
        ("rear_below", (1., 1.6, .55), (1., .15, 1.1)),
        ("rear_above", (1., 1.6, 2.525), (1., .15, .55)),
        ("front_left", (-.65, -2., 1.4), (2.6, .15, 2.8)),
        ("front_right", (1.80, -2., 1.4), (.3, .15, 2.8)),
        ("front_lintel", (1.15, -2., 2.45), (1., .15, .7))):
        lab.box(name, position, size, "wall")
    lab.box("window_glass", (1., 1.6, 1.675), (.944, .004, 1.094), "glass")
    for x in (.51, 1.0, 1.49):
        lab.box(f"window_vertical_{x}", (x, 1.577, 1.675), (.025, .06, 1.15), "cabinet")
    for z in (1.11, 2.24):
        lab.box(f"window_horizontal_{z}", (1., 1.577, z), (1., .06, .025), "cabinet")
    # Closed reagent storage is on the wall, clear of the robot work surface.
    for x in (-1.25, -.55):
        lab.box(f"wall_cabinet_{x}", (x, 1.30, 1.65), (.64, .45, .76), "cabinet")
        for side in (-1, 1):
            dx = x+side*.155
            lab.box(f"wall_cabinet_door_{dx}", (dx, 1.064, 1.65), (.301, .018, .734), "cabinet")
            lab.cylinder(f"wall_cabinet_handle_{dx}", (x+side*.028, 1.039, 1.60),
                .004, .12, "steel")
    # Coved wall/floor skirting: sealed room finish, no exposed porous wood.
    lab.box("rear_skirting", (0., 1.515, .055), (3.65, .018, .11), "skirting")
    lab.box("left_skirting", (-1.815, -.2, .055), (.018, 3.43, .11), "skirting")
    lab.box("right_skirting", (1.815, -.2, .055), (.018, 3.43, .11), "skirting")
    # Ceiling light housings match the emitters' positions, outside the
    # workcell's reachable region. They are not floating rectangle lights.
    for index, (x, y) in enumerate(((-.6, -.1), (.6, .4))):
        lab.box(f"luminaire_{index}", (x, y, 2.677), (.33, 1.23, .052), "cabinet")
        lab.box(f"luminaire_diffuser_{index}", (x, y, 2.651), (.30, 1.20, .002), "ptfe")

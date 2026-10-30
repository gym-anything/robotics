"""Everything the renderer draws except the truck and the robot (render.py owns those, the lights and the cameras).

  interior.py   the building from inside: floor, walls, roof structure, columns, racks and what is stored in them, the
                conveyor and press, painted markings, signs and dressing (warehouse and dock presets)
  dockyard.py   the dock wall with its doors, seals, bumpers and levelers, the trailers, and the whole yard preset
  loads.py      what the truck handles: pallets, wrapped unit loads, loose cartons, coils, bales, dunnage
  common.py     which module draws which layout part, and shared readers

Every module exposes draw(S, world): S is forklift.usd.Stage (the stage being exported), world the built World. Bodies
are drawn under /Lab/body_<index> in the body's frame (the render worker moves those prims each frame); everything
static under /World in world coordinates.
"""


def draw(S, world):
    from . import dockyard, interior, loads
    if world.scene.layout is None and hasattr(dockyard, "draw_lab"):
        dockyard.draw_lab(S, world)          # the experiments' outdoor test pad (preset lab)
    else:
        interior.draw(S, world)
        dockyard.draw(S, world)
    loads.draw(S, world)

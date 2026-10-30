"""Metric folding workstation. Visible furniture is also collision geometry."""
CAMERA_EYE = (1.65, -1.65, 2.15)
CAMERA_TARGET = (0., 0., .8)


def build(builder, c, newton, wp):
    materials = {}
    def box(name, center, half, material="wood", mu=.4):
        i = builder.add_shape_box(-1, xform=wp.transform(center, wp.quat_identity()),
            hx=half[0], hy=half[1], hz=half[2], label=name,
            cfg=newton.ModelBuilder.ShapeConfig(mu=mu, ke=1e7, kd=1000., gap=.0003))
        materials[i] = material
        return i
    box("room_floor", (0, 0, -.025), (2.8, 2.8, .025), "floor")
    # The actual cloth contact tabletop is supplied by the retained adapter.
    for x in (-c.table_half_x+.06, c.table_half_x-.06):
        for y in (-c.table_half_y+.06, c.table_half_y-.06):
            box(f"table_leg_{x}_{y}", (x, y, (c.table_top-.08)/2), (.03, .03, (c.table_top-.08)/2), "steel")
    for sign in (-1, 1):
        box(f"robot_pedestal_{sign}", (0, sign*c.robot_base_y, (c.table_top-.04)/2),
            (.11, .11, (c.table_top-.04)/2), "steel")
        box(f"robot_mount_plate_{sign}", (0, sign*c.robot_base_y, c.table_top-.02), (.15, .15, .02), "aluminum")
    box("back_wall", (-2.2, 0, 1.4), (.05, 2.8, 1.4), "wall")
    box("side_wall", (0, 2.5, 1.4), (2.2, .05, 1.4), "wall")
    box("opposite_wall", (0, -2.5, 1.4), (2.2, .05, 1.4), "wall")
    # Open shelves; no fake folded garments masquerading as physical inventory.
    for z in (.10, .55, 1., 1.45):
        box(f"shelf_{z}", (-1.78, .55, z), (.28, .8, .018), "wood")
    for y in (-.23, 1.33):
        box(f"shelf_side_{y}", (-1.78, y, .78), (.28, .018, .78), "cabinet")
    for name, x, y in (("empty_basket", 1.27, .58), ("spare_basket", -1.76, -.93)):
        z, hx, hy, height = .035, .23, .18, .32
        box(name+"_base", (x, y, z), (hx, hy, .015), "basket")
        for sign in (-1, 1):
            box(name+f"_end_{sign}", (x+sign*hx, y, z+height/2), (.008, hy, height/2), "basket")
            box(name+f"_side_{sign}", (x, y+sign*hy, z+height/2), (hx, .008, height/2), "basket")
    # External camera stand, outside the robot workspace, with physical base.
    # Place the stand behind the optical centre, not through the viewing ray.
    import numpy as np
    direction = np.asarray(CAMERA_TARGET)-np.asarray(CAMERA_EYE)
    direction /= np.linalg.norm(direction)
    x, y, z = np.asarray(CAMERA_EYE)-.10*direction
    box("camera_foot", (x, y, .02), (.19, .19, .02), "steel")
    box("camera_pole", (x, y, z/2), (.016, .016, z/2), "steel")
    box("camera_housing", (x, y, z), (.055, .035, .03), "robot_dark")
    return materials

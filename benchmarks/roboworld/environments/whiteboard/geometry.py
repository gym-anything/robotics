"""One scene description consumed by Newton and USD. Dynamic objects share poses."""

from dataclasses import dataclass, field
import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Part:
    name: str
    kind: str
    size: tuple
    position: tuple = (0, 0, 0)
    rotation: tuple = (0, 0, 0, 1)
    material: str = "wall"
    collision: bool = True
    friction: float = 0.4
    bevel: float = 0.001
    vertices: object = None
    faces: object = None


@dataclass
class Object:
    name: str
    position: tuple = (0, 0, 0)
    rotation: tuple = (0, 0, 0, 1)
    mass: float = 0.0
    parts: list = field(default_factory=list)

    def box(self, name, size, position=(0, 0, 0), material="wall", **kwargs):
        self.parts.append(
            Part(name, "box", tuple(size), tuple(position), material=material, **kwargs)
        )
        return self

    def cylinder(self, name, radius, length, position=(0, 0, 0), material="steel", **kwargs):
        self.parts.append(
            Part(
                name,
                "cylinder",
                (radius, length),
                tuple(position),
                material=material,
                **kwargs,
            )
        )
        return self

    def tube(self, name, start, end, radius=0.0127, material="steel", **kwargs):
        a, b = np.array(start), np.array(end)
        d = b - a
        rot, _ = Rotation.align_vectors([d / np.linalg.norm(d)], [[0, 0, 1]])
        return self.cylinder(
            name,
            radius,
            float(np.linalg.norm(d)),
            (a + b) / 2,
            material,
            rotation=tuple(rot.as_quat()),
            **kwargs,
        )

    def curved_back(self):
        # Closed, continuously curved polyethylene shell. Shared mesh source
        # for appearance and collision; MuJoCo uses its convex collision hull.
        nx, nz = 18, 16
        vertices = []
        for side in (-1, 1):
            for j in range(nz):
                z = (j / (nz - 1) - 0.5) * 0.285
                taper = 1 - 0.055 * (j / (nz - 1)) ** 3
                for i in range(nx):
                    x = (i / (nx - 1) - 0.5) * 0.416 * taper
                    y = -0.024 * (x / 0.208) ** 2 + 0.25 * z + side * 0.003
                    vertices.append((x, y, z))
        faces = []
        layer = nx * nz
        for j in range(nz - 1):
            for i in range(nx - 1):
                a = j * nx + i
                b = a + 1
                c = a + nx
                d = c + 1
                faces.extend(
                    (
                        (a, c, b),
                        (b, c, d),
                        (a + layer, b + layer, c + layer),
                        (b + layer, d + layer, c + layer),
                    )
                )
        boundary = (
            list(range(nx))
            + [j * nx + nx - 1 for j in range(1, nz)]
            + list(range(layer - 2, layer - nx - 1, -1))
            + [j * nx for j in range(nz - 2, 0, -1)]
        )
        for a, b in zip(boundary, boundary[1:] + boundary[:1]):
            faces.extend(((a, b, a + layer), (b, b + layer, a + layer)))
        self.parts.append(
            Part(
                "back_shell",
                "mesh",
                (),
                (0, 0.222, 0.625),
                material="chair_blue",
                vertices=np.asarray(vertices),
                faces=np.asarray(faces)[:, ::-1],
            )
        )
        return self


def classroom(config):
    b = config.board
    objects = []
    room = Object("architecture")
    room.box("floor", (7.4, 8.2, 0.12), (0, 4, -0.06), "linoleum", friction=0.7)
    room.box("front_wall", (7.4, 0.15, 3.25), (0, -0.095, 1.625), "wall")
    room.box("right_wall", (0.15, 8.2, 3.25), (3.675, 4, 1.625), "wall")
    room.box("back_wall", (7.4, 0.15, 3.25), (0, 8.075, 1.625), "wall")
    room.box("ceiling", (7.4, 8.2, 0.12), (0, 4, 3.31), "ceiling")
    # Real window openings: wall is split; no opaque wall behind glass.
    room.box("window_sill_wall", (0.15, 8.2, 0.90), (-3.675, 4, 0.45), "wall")
    room.box("window_header", (0.15, 8.2, 0.65), (-3.675, 4, 2.925), "wall")
    for i, y in enumerate((0, 2.65, 5.3, 8.0)):
        room.box(f"window_pier_{i}", (0.15, 0.24, 1.70), (-3.675, y, 1.75), "wall")
    for i, y in enumerate((1.325, 3.975, 6.65)):
        room.box(f"window_glass_{i}", (0.008, 2.38, 1.64), (-3.68, y, 1.75), "glass")
        for j, z in enumerate((0.91, 2.59)):
            room.box(f"sill_{i}_{j}", (0.21, 2.55, 0.045), (-3.60, y, z), "paint_white")
        room.box(f"mullion_{i}", (0.05, 0.035, 1.65), (-3.57, y, 1.75), "paint_white")
    room.box("front_skirting", (7.2, 0.016, 0.10), (0, 0.003, 0.05), "charcoal")
    room.box("right_skirting", (0.016, 8, 0.10), (3.59, 4, 0.05), "charcoal")
    # Door on front wall, framed leaf; closed architectural reference.
    room.box("door_leaf", (0.93, 0.042, 2.08), (2.65, 0.022, 1.04), "wood")
    for i, x in enumerate((2.15, 3.15)):
        room.box(f"door_jamb_{i}", (0.065, 0.080, 2.18), (x, 0.025, 1.09), "paint_white")
    room.box("door_header", (1.065, 0.080, 0.065), (2.65, 0.025, 2.15), "paint_white")
    room.box("door_window", (0.23, 0.008, 0.56), (2.38, 0.048, 1.56), "dark_glass")
    room.tube("door_handle", (2.99, 0.09, 1.0), (2.86, 0.09, 1.0), 0.012, "chrome")
    for i, y in enumerate((1.6, 4, 6.5)):
        for j, x in enumerate((-1.6, 1.6)):
            room.box(
                f"light_housing_{i}_{j}",
                (0.34, 1.2, 0.065),
                (x, y, 3.18),
                "paint_white",
            )
            room.box(
                f"light_diffuser_{i}_{j}",
                (0.30, 1.15, 0.010),
                (x, y, 3.142),
                "diffuser",
            )
    # Acoustic ceiling grid and fine expansion lines on linoleum.
    for i in range(13):
        room.box(
            f"ceiling_grid_x_{i}",
            (0.012, 8.0, 0.012),
            (-3.6 + i * 0.6, 4, 3.245),
            "paint_white",
            collision=False,
        )
    for i in range(14):
        room.box(
            f"ceiling_grid_y_{i}",
            (7.2, 0.012, 0.012),
            (0, i * 0.6, 3.245),
            "paint_white",
            collision=False,
        )
    objects.append(room)

    board = Object("board")
    board.box(
        "mdf_backer",
        (b.width - 0.01, b.backer, b.height - 0.01),
        (0, b.face_y - b.backer / 2 - 0.001, b.center_z),
        "mdf",
    )
    board.box(
        "porcelain",
        (b.writing_width, 0.001, b.writing_height),
        (0, b.face_y - 0.0005, b.center_z),
        "board",
        friction=0.35,
    )
    for i, x in enumerate((-b.width / 2 + b.trim / 2, b.width / 2 - b.trim / 2)):
        board.box(
            f"trim_vertical_{i}",
            (b.trim, 0.025, b.height),
            (x, b.face_y - 0.003, b.center_z),
            "aluminum",
            bevel=0.003,
        )
    for i, z in enumerate((b.bottom + b.trim / 2, b.bottom + b.height - b.trim / 2)):
        board.box(
            f"trim_horizontal_{i}",
            (b.width, 0.025, b.trim),
            (0, b.face_y - 0.003, z),
            "aluminum",
            bevel=0.003,
        )
    # Tray projection 80 mm is an explicitly declared geometric prior.
    tray_z = b.bottom - 0.015
    board.box("tray_floor", (b.width, 0.080, 0.003), (0, b.face_y + 0.037, tray_z), "aluminum")
    board.box(
        "tray_lip",
        (b.width, 0.003, 0.013),
        (0, b.face_y + 0.077, tray_z + 0.005),
        "aluminum",
    )
    for i, x in enumerate((-b.width / 2, b.width / 2)):
        board.box(
            f"tray_endcap_{i}",
            (0.008, 0.085, 0.025),
            (x, b.face_y + 0.037, tray_z + 0.004),
            "charcoal",
            bevel=0.003,
        )
    for i, x in enumerate((-0.85, 0.85)):
        board.box(
            f"tray_bracket_{i}",
            (0.025, 0.063, 0.008),
            (x, b.face_y + 0.026, tray_z - 0.008),
            "aluminum",
        )
        board.box(f"mount_spacer_{i}", (0.060, 0.058, 0.13), (x, 0.046, b.center_z), "steel")
    objects.append(board)

    if config.furniture:
        for row, y in enumerate((3.25, 4.9, 6.55)):
            for col, x in enumerate((-2.05, -0.7, 0.7, 2.05)):
                desk = Object(f"desk_{row}_{col}", (x, y, 0), mass=config.materials.desk_mass_kg)
                desk.box(
                    "top",
                    (0.6096, 0.4572, 0.025),
                    (0, 0, 0.7241),
                    "laminate",
                    bevel=0.010,
                )
                desk.box(
                    "bookbox_bottom",
                    (0.5334, 0.4064, 0.004),
                    (0, 0, 0.594),
                    "dark_plastic",
                )
                for j, sx in enumerate((-0.269, 0.269)):
                    desk.box(
                        f"bookbox_side_{j}",
                        (0.004, 0.4064, 0.104),
                        (sx, 0, 0.644),
                        "dark_plastic",
                    )
                desk.box(
                    "bookbox_back",
                    (0.5334, 0.004, 0.104),
                    (0, -0.202, 0.644),
                    "dark_plastic",
                )
                for j, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
                    desk.tube(
                        f"leg_{j}",
                        (sx * 0.275, sy * 0.19, 0.025),
                        (sx * 0.249, sy * 0.17, 0.705),
                        0.014,
                    )
                    desk.cylinder(
                        f"glide_{j}",
                        0.020,
                        0.013,
                        (sx * 0.275, sy * 0.19, 0.007),
                        "charcoal",
                        friction=0.30,
                    )
                    desk.cylinder(
                        f"leg_collar_{j}",
                        0.0155,
                        0.025,
                        (sx * 0.262, sy * 0.18, 0.36),
                        "chrome",
                        collision=False,
                    )
                objects.append(desk)
                chair = Object(
                    f"chair_{row}_{col}",
                    (x + 0.025 * (col % 2), y + 0.48, 0),
                    mass=config.materials.chair_mass_kg,
                )
                # Thin seat with support rails and a continuous curved back shell.
                chair.box(
                    "seat",
                    (0.415, 0.41, 0.006),
                    (0, 0, 0.4542),
                    "chair_blue",
                    bevel=0.004,
                )
                chair.curved_back()
                for j, sy in enumerate((-0.155, 0.155)):
                    chair.tube(f"seat_rail_{j}", (-0.18, sy, 0.439), (0.18, sy, 0.439), 0.0127)
                for j, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
                    chair.tube(
                        f"leg_{j}",
                        (sx * 0.218, sy * 0.217, 0.016),
                        (sx * 0.177, sy * 0.162, 0.43),
                        0.0127,
                    )
                    chair.cylinder(
                        f"glide_{j}",
                        0.016,
                        0.012,
                        (sx * 0.218, sy * 0.217, 0.006),
                        "charcoal",
                        friction=0.28,
                    )
                for j, sx in enumerate((-0.18, 0.18)):
                    chair.tube(f"back_support_{j}", (sx, 0.15, 0.33), (sx, 0.24, 0.75), 0.011)
                objects.append(chair)
        teacher = Object("teacher_desk", (-2.35, 1.20, 0), mass=28.0)
        teacher.box("top", (1.20, 0.65, 0.030), (0, 0, 0.735), "wood", bevel=0.008)
        teacher.box("modesty_panel", (1.06, 0.02, 0.43), (0, -0.23, 0.43), "paint_white")
        for i, (x, y) in enumerate(((-0.53, -0.26), (-0.53, 0.26), (0.53, -0.26), (0.53, 0.26))):
            teacher.box(f"leg_{i}", (0.035, 0.035, 0.72), (x, y, 0.36), "steel")
        objects.append(teacher)
        teacher_chair = copy.deepcopy(next(obj for obj in objects if obj.name == "chair_0_0"))
        teacher_chair.name = "teacher_chair"
        teacher_chair.position = (-2.35, 0.65, 0)
        teacher_chair.rotation = tuple(Rotation.from_euler("z", math.pi).as_quat())
        objects.append(teacher_chair)
    for obj in objects:
        if obj.name.startswith(("desk_", "chair_")):
            x, y, z = obj.position
            obj.position = (x + config.layout.furniture_offset_x, y, z)
    return objects


def marker(color, position):
    # Axis local +Z points towards nib; cap is its own hollow dynamic object.
    rot = tuple(Rotation.from_euler("y", math.pi / 2).as_quat())
    barrel = Object(f"marker_{color}", tuple(position), rot, 0.0162)
    barrel.cylinder("barrel", 0.007, 0.116, (0, 0, -0.007), "marker_white", friction=0.45)
    barrel.cylinder("collar", 0.007, 0.011, (0, 0, 0.050), color, friction=0.45)
    barrel.cylinder("tail", 0.007, 0.006, (0, 0, -0.062), color)
    barrel.box(
        "nib",
        (0.005, 0.002, 0.0075),
        (0, 0, 0.05925),
        color,
        friction=0.35,
        bevel=0.0004,
    )
    cap_pos = np.asarray(position) + Rotation.from_quat(rot).apply((0, 0, 0.0555))
    cap = Object(f"cap_{color}", tuple(cap_pos), rot, 0.0034)
    # Outer rigid proxies for finger/furniture collision only. The rendered and
    # barrel-contact wall is an elastic tetrahedral shell with a clearance bore
    # and three retention ribs (see elasticity.py). No cap-parent joint.
    for i in range(12):
        angle = i * math.tau / 12
        cap.box(
            f"wall_{i}",
            (0.0041, 0.00095, 0.035),
            (0.008025 * math.sin(angle), 0.008025 * math.cos(angle), 0),
            color,
            rotation=tuple(Rotation.from_euler("z", -angle).as_quat()),
            bevel=0.0003,
        )
    cap.cylinder("end", 0.0085, 0.0014, (0, 0, 0.0168), color)
    cap.box("clip", (0.003, 0.0015, 0.028), (0, 0.0077, -0.003), color, bevel=0.0005)
    return barrel, cap


def tools(config):
    b = config.board
    rng = np.random.default_rng(config.seed)
    objects = []
    # Ordinary horizontal storage across a full-length tray, capped initially.
    for i, color in enumerate(config.layout.marker_order):
        objects.extend(
            marker(
                color,
                (
                    -0.92 + i * 0.172 + config.layout.marker_offset_x,
                    b.face_y + 0.036 + float(rng.uniform(-0.004, 0.004)),
                    b.bottom - 0.015 + 0.011,
                ),
            )
        )
    eraser = Object(
        "eraser",
        (config.layout.eraser_x, b.face_y + 0.041, b.bottom - 0.015 + 0.012),
        mass=config.materials.eraser_mass_kg,
    )
    eraser.box("housing", (0.107, 0.057, 0.015), (0, 0, 0.0025), "eraser_blue", bevel=0.005)
    eraser.box(
        "pad",
        (0.103, 0.053, 0.005),
        (0, 0, -0.0075),
        "felt",
        friction=0.55,
        bevel=0.001,
    )
    objects.append(eraser)
    return objects

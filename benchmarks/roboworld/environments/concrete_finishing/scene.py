"""Physics scene of mechanical room B-012 (Newton 1.5 ModelBuilder): the installed room, the locked tool cart and the
station props as static parts, the hand tools as free rigid bodies, the catalog robot, and one support plate per
contactor (every tool sole and every fingertip pad).

Reuses classroom_arrangement.scene.Scene (the qualified robot mount, fingertip pads, cameras and gravity-compensation
set-up) and replaces what is room-specific. The wet materials are not colliders: they act through the plates
(coupling.py), which are static boxes parked out of the room until the world moves one under its sole.
"""

import math
import numpy as np
from classroom_arrangement.scene import Scene
from . import params as P
from . import tools as T
from .coupling import PLATE_HALF_T

PARK = (-20.0, -20.0, -5.0)              # where idle plates wait (out of the room, under the floor)
PLATE_MARGIN = 0.004                     # a plate is this much larger than its sole on every side
PAD_TIP_Z = 0.04525 + 0.00925            # the fingertip pad's outer face in the finger frame (franka_description)
PAD_HALF = (0.00875, 0.0076)             # the pad's face, half extents across the finger and along its width


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw/2), math.cos(yaw/2))


class FinishScene(Scene):
    def __init__(self, config, newton, wp, spec):
        self.tools = {}                  # name -> dict(tool, body, shapes, sole_shapes)
        self.static_shapes = []
        self.part_of_shape = {}
        self.plate_shapes = {}           # contactor name -> plate shape
        self.contactor_specs = []        # dicts the world turns into coupling.Contactor
        self.side_specs = []             # and the tools' side contactors, each on its tool's plate
        super().__init__(config, newton, wp, spec)
        self.tooling()
        self.make_plates()

    # ------------------------------------------------------------------ room
    def room(self):
        b, wp = self.b, self.wp
        b.add_ground_plane(cfg=self.cfg(mu=P.MU_FLOOR), label="floor")
        self.floor_shape = b.shape_count - 1
        self.materials[self.floor_shape] = "floor"
        for name, c, h, mat in self.spec["statics"]:
            i = b.add_shape_box(-1, xform=wp.transform(c, wp.quat_identity()), hx=h[0], hy=h[1], hz=h[2],
                                cfg=self.cfg(mu=0.5, density=0.), label=name)
            self.materials[i] = mat
            self.static_shapes.append(i)
        for name, c, h, yaw, mat in self.spec["cart_boxes"]:
            i = b.add_shape_box(-1, xform=wp.transform(c, wp.quat(*yaw_quat(yaw))), hx=h[0], hy=h[1], hz=h[2],
                                cfg=self.cfg(mu=0.4, density=0.), label=name)
            self.materials[i] = mat
            self.static_shapes.append(i)

    # ------------------------------------------------------------------ robot
    def robot(self):
        """The classroom scene's robot set-up with this environment's catalog entry (default robot and supported
        configurations: `concrete-finishing` in robots/catalog.py)."""
        import robots.catalog as catalog
        original = catalog.for_environment
        catalog.for_environment = lambda env, selector: original("concrete-finishing", selector)
        try:
            super().robot()
        finally:
            catalog.for_environment = original

    def furniture(self):
        """No furniture: the room's fixed parts are statics and the tools are added by tooling()."""

    # ------------------------------------------------------------------ tools
    def tooling(self):
        wp, b = self.wp, self.b
        for name, (pos, rot) in self.spec["tools"].items():
            tool = T.make(name)
            quat = tuple(rot) if isinstance(rot, (tuple, list)) else yaw_quat(rot)     # a yaw, or (probes) a quaternion
            body = b.add_body(xform=wp.transform(pos, wp.quat(*quat)), label=name)
            shapes, sole = [], []
            for p in tool.parts:
                s = self.add_part(body, p, name)
                if s is None:
                    continue
                shapes.append(s)
                self.part_of_shape[s] = (name, p)
                if p.name == tool.sole_part:
                    sole.append(s)
            self.bodies[name] = body
            self.parts[name] = tool.parts
            self.items[name] = dict(kind="tool", body=body, shapes=shapes, glides=[])
            self.tools[name] = dict(tool=tool, body=body, shapes=shapes, sole_shapes=sole)

    # ------------------------------------------------------------------ support plates
    def make_plates(self):
        """One frictionless static box per contactor, parked; the world moves it under its sole every tick."""
        wp, b = self.wp, self.b
        specs = []
        self.side_specs = []             # each tool's other sides: a second contactor on the same plate (coupling)
        for name, t in self.tools.items():
            tool = t["tool"]
            sides = T.side_faces(tool)
            # the plate is large enough for whichever side bears on it (a knife on its back spans its handle)
            hx = max([(tool.bear or tool.sole).hx] + [f["sole"].hx for f in sides])
            hy = max([(tool.bear or tool.sole).hy] + [f["sole"].hy for f in sides])
            specs.append(dict(name=name, body=t["body"], p_ls=(0.0, 0.0, 0.0), R_ls=np.eye(3), sole=tool.sole,
                              mass=tool.mass, tool=tool, sole_shapes=list(t["sole_shapes"]), bear=tool.bear,
                              half=(hx + PLATE_MARGIN, hy + PLATE_MARGIN)))
            self.side_specs.append(dict(name=f"{name}:side", main=name, body=t["body"], faces=sides, mass=tool.mass,
                                        shapes=list(t["shapes"])))
        # fingertip pads: the sole is the pad's outer face, facing out of the finger (sole frame z back into it)
        from .paste import Sole
        Rflip = np.diag([1.0, -1.0, -1.0])
        for s, label in self.pad_shapes.items():
            body = int(b.shape_body[s])
            short = label.split("/")[-1]
            specs.append(dict(name=f"pad:{short}", body=body, p_ls=(0.0, 0.0, PAD_TIP_Z), R_ls=Rflip,
                              sole=Sole("blade", *PAD_HALF), mass=0.6, tool=None, sole_shapes=[s],
                              half=(PAD_HALF[0] + PLATE_MARGIN, PAD_HALF[1] + PLATE_MARGIN)))
        for k, sp in enumerate(specs):
            pos = (PARK[0] - 0.5*k, PARK[1], PARK[2])
            hx, hy = sp["half"]
            i = b.add_shape_box(-1, xform=wp.transform(pos, wp.quat_identity()), hx=hx, hy=hy, hz=PLATE_HALF_T,
                                cfg=self.cfg(mu=0.01, density=0.), label=f"plate/{sp['name']}")
            self.materials[i] = "plate"
            self.plate_shapes[sp["name"]] = i
            sp["plate_shape"] = i
            sp["park"] = pos
        self.contactor_specs = specs

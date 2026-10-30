"""Newton rigid laboratory apparatus built from physical cavities.

Bodies are free unless actually clamped/mounted. No grasp welds, liquid
attachments, or prescribed stir-bar motion are supplied by this module.
"""

from dataclasses import dataclass, field
import numpy as np
from .geometry import flask, profile_wall_cells, FLASK_PROFILE, record_meridian
from .runtime import require_newton


@dataclass
class Apparatus:
    builder: object
    bodies: dict = field(default_factory=dict)
    visuals: dict = field(default_factory=dict)
    joints: dict = field(default_factory=dict)

    def add_flask(self, name="flask", position=(0., 0., .802)):
        newton, wp = require_newton()
        body = self.builder.add_body(xform=wp.transform(position, wp.quat_identity()), label=name)
        mesh = flask()
        # Whole hollow mesh: fluid collisions + glass optics. Its density
        # supplies mass once; the rigid convex cells have zero extra density.
        shape = self.builder.add_shape_mesh(body, mesh=newton.Mesh(mesh.vertices, mesh.triangles.flatten()),
                                    cfg=self.builder.ShapeConfig(density=2230., mu=.25,
                                                                  has_shape_collision=False), label=name+"_glass")
        record_meridian(self.builder, shape, mesh)
        for index, cell in enumerate(profile_wall_cells(FLASK_PROFILE)):
            self.builder.add_shape_convex_hull(body,
                mesh=newton.Mesh(cell.vertices, cell.triangles.flatten()),
                cfg=self.builder.ShapeConfig(density=0., has_particle_collision=False,
                                              is_visible=False, mu=.25), label=f"{name}_wall_{index}")
        self.bodies[name] = body
        self.visuals[name] = mesh
        return body

    def add_stir_bar(self, position=(0., 0., .809), name="stir_bar"):
        _, wp = require_newton()
        body = self.builder.add_body(xform=wp.transform(position, wp.quat_identity()), label=name)
        # PTFE-sheathed 20 x 6 mm bar. Density 3200 kg/m3 estimates the mixed
        # PTFE + magnetic core bulk density; no angle/position trajectory.
        self.builder.add_shape_capsule(body, radius=.003, half_height=.007,
                                       xform=wp.transform((0., 0., 0.), wp.quat_from_axis_angle(wp.vec3(0., 1., 0.), np.pi/2)),
                                       cfg=self.builder.ShapeConfig(density=3200., mu=.15), label=name)
        self.bodies[name] = body
        return body


def build_flask_fixture(device="cuda:0", *, stir_bar=False):
    from .rigid_contact import register, configure

    newton, wp = require_newton()
    builder = newton.ModelBuilder()
    register(builder)
    builder.default_shape_cfg.gap = .0002
    builder.add_shape_box(-1, hx=.2, hy=.2, hz=.02,
                         xform=wp.transform((0., 0., -.02), wp.quat_identity()),
                         cfg=builder.ShapeConfig(mu=.5), label="bench")
    apparatus = Apparatus(builder)
    apparatus.add_flask(position=(0., 0., .0005))
    if stir_bar:
        apparatus.add_stir_bar(position=(0., 0., .009))
    model = builder.finalize(device=device)
    configure(model)
    model.set_gravity((0., 0., -9.80665))
    return apparatus, model

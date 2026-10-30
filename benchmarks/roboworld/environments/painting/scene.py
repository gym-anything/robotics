"""Physics scene (Newton 1.5 ModelBuilder): the bedroom's installed parts as static colliders, the Mobile FR3 Duo (the
classroom_arrangement robot stack, unchanged) and the preset's tools and props as free rigid bodies.

Articulated tools: the roller cover turns on the frame's cage axle (revolute joint with bearing friction); the aerosol
actuator button rides on a 2.2 mm slide joint whose spring (joint position servo with zero target velocity) holds it up
with the valve spring's preload. Everything else is a single rigid body; every part's density is its mass share over
its volume (classroom Scene.add_part), so masses, centres of mass and inertias follow the geometry.
"""

import math
import numpy as np

from classroom_arrangement.scene import Scene as ClassroomScene
from classroom_arrangement import params as CP
from . import params as P
from . import layout as L
from . import tools as T

BUILDERS = dict(tray=T.tray, pail=T.pail, can=T.can, palette=T.palette, table=T.table, board=T.board,
                spray_box=T.spray_box, crate=T.crate_side, stencil=T.stencil, sash=lambda: T.brush("sash"),
                round8=lambda: T.brush("round8"), stencil_brush=lambda: T.brush("stencil"),
                longreach=lambda: T.brush("longreach"))
MU = dict(tray=0.35, pail=0.35, can=0.30, palette=0.35, table=0.40, board=0.40, spray_box=0.45, crate=0.45, stencil=0.35,
          sash=0.40, round8=0.40, longreach=0.40, roller=0.40, aerosol=0.35)
PARKED = (0.0, -1.2, 0.3)          # held tools are built here and placed into the hand by the environment


class PaintingScene(ClassroomScene):
    ENV_ID = "painting"

    def __init__(self, config, newton, wp, spec):
        self.c, self.newton, self.wp, self.spec = config, newton, wp, spec
        newton.use_coord_layout_targets = True
        self.b = newton.ModelBuilder(gravity=(0., 0., -P.GRAVITY))
        from newton.solvers import SolverMuJoCo
        SolverMuJoCo.register_custom_attributes(self.b)
        self.materials, self.parts = {}, {}
        self.bodies, self.items = {}, {}
        self.glide_shapes, self.wheel_shapes, self.pad_shapes = [], [], {}
        self.cameras = {}
        self.static_shapes = {}
        self.joints_extra = {}
        self.room()
        self.robot()
        self.tools()

    # ------------------------------------------------------------------ room
    def room(self):
        b, wp = self.b, self.wp
        b.add_ground_plane(cfg=self.cfg(mu=0.45), label="floor")
        self.materials[b.shape_count - 1] = "floor"
        self.static_shapes["floor"] = b.shape_count - 1
        for name, c, h, mat in L.static_boxes():
            i = b.add_shape_box(-1, xform=wp.transform(c, wp.quat_identity()), hx=h[0], hy=h[1], hz=h[2],
                                cfg=self.cfg(mu=0.45, density=0.), label=name)
            self.materials[i] = mat
            self.static_shapes[name] = i
        pc = L.PIPE
        i = b.add_shape_cylinder(-1, xform=wp.transform((pc["x"], pc["y"], pc["z1"]/2), wp.quat_identity()), radius=pc["r"],
                                 half_height=pc["z1"]/2, cfg=self.cfg(mu=0.4, density=0.), label="rad_pipe")
        self.materials[i] = "radiator"
        self.static_shapes["rad_pipe"] = i

    # ------------------------------------------------------------------ tools
    def _body(self, name, kind, pos, quat, parts):
        wp = self.wp
        body = self.b.add_body(xform=wp.transform(pos, wp.quat(*quat)), label=name)
        shapes = [self._part(body, p, name, MU.get(kind)) for p in parts]
        self.bodies[name] = body
        self.parts[name] = parts
        self.items[name] = dict(kind=kind, body=body, shapes=[s for s in shapes if s is not None], glides=[])
        return body

    def _part(self, body, p, prefix, mu):
        if p.mu is None and mu is not None:
            p.mu = mu
        i = self.add_part(body, p, prefix)
        if i is None and p.mass > 0:
            # drawn-only part with mass (bristles, cage axle): add its mass and inertia at its place
            from classroom_arrangement.scene import volume
            wp = self.wp
            i = self.b.add_shape_box(body, xform=wp.transform(p.pos, wp.quat(*p.quat)), hx=p.size[0],
                                     hy=p.size[1] if p.kind == "box" else p.size[0], hz=p.size[2] if p.kind == "box" else p.size[1],
                                     cfg=self.cfg(density=p.mass/volume(p) if p.kind == "box" else
                                                  p.mass/(8*p.size[0]*p.size[0]*p.size[1]), has_shape_collision=False,
                                                  has_particle_collision=False, is_visible=False), label=f"{prefix}/{p.name}_mass")
            self.materials[i] = "mass_only"
        return i

    def tools(self):
        sp = self.spec
        held = set(sp["held"].values())
        for name, it in sp["items"].items():
            x, y, z, q = it["pose"]
            kind = it["kind"]
            if kind == "roller":
                self.roller(name, (x, y, z), q)
            elif kind == "aerosol":
                self.aerosol(name, (x, y, z), q)
            else:
                self._body(name, kind, (x, y, z + 0.0005), q, BUILDERS[kind]())
        for k, name in enumerate(sorted(held)):
            pos = (PARKED[0] + 0.4*k, PARKED[1], PARKED[2])
            if name == "roller":
                self.roller(name, pos, (0., 0., 0., 1.))
            elif name == "aerosol":
                self.aerosol(name, pos, (0., 0., 0., 1.))
            else:
                self._body(name, name, pos, (0., 0., 0., 1.), BUILDERS[name]())

    def roller(self, name, pos, quat):
        """Frame (free) + cover on the cage axle (revolute, bearing friction)."""
        b, wp = self.b, self.wp
        frame = b.add_link(xform=wp.transform(pos, wp.quat(*quat)), label=name)
        jf = b.add_joint_free(frame, label=f"{name}_free_joint")
        fparts = T.roller_frame()
        shapes = [self._part(frame, p, name, MU["roller"]) for p in fparts]
        cc = np.asarray(T.roller_cover_center())
        from scipy.spatial.transform import Rotation
        Rq = Rotation.from_quat(quat)
        cpos = np.asarray(pos) + Rq.apply(cc)
        cover = b.add_link(xform=wp.transform(tuple(cpos), wp.quat(*quat)), label=f"{name}_cover")
        cparts = T.roller_cover()
        cshapes = [self._part(cover, p, f"{name}_cover", MU["roller"]) for p in cparts]
        jr = b.add_joint_revolute(frame, cover, parent_xform=wp.transform(tuple(cc), wp.quat_identity()),
                                  child_xform=wp.transform((0., 0., 0.), wp.quat_identity()), axis=(0., 1., 0.),
                                  target_ke=0.0, target_kd=0.0, friction=P.ROLLER["bearing"], armature=0.0,
                                  label=f"{name}_cover_joint")
        b.add_articulation([jf, jr], label=f"{name}_articulation")
        self.bodies[name], self.bodies[f"{name}_cover"] = frame, cover
        self.parts[name], self.parts[f"{name}_cover"] = fparts, cparts
        self.items[name] = dict(kind="roller", body=frame, shapes=[s for s in shapes if s is not None], glides=[])
        self.items[f"{name}_cover"] = dict(kind="roller_cover", body=cover, shapes=[s for s in cshapes if s is not None],
                                           glides=[])
        self.joints_extra[f"{name}_cover"] = jr

    def aerosol(self, name, pos, quat):
        """Can (free) + actuator button on a slide joint along the can axis, sprung up against its top stop."""
        b, wp = self.b, self.wp
        A = P.AEROSOL
        can = b.add_link(xform=wp.transform(pos, wp.quat(*quat)), label=name)
        jf = b.add_joint_free(can, label=f"{name}_free_joint")
        cparts = T.aerosol_can()
        # the liquid fill is lumped into the can: a drawn-only mass part over the liquid's height
        from classroom_arrangement.furniture import Part
        cparts.append(Part("fill", "cylinder", (A["r"]*0.95, 0.06), (0., 0., 0.065), mass=A["fill_mass"], material="fill",
                           collide=False))
        shapes = [self._part(can, p, name, MU["aerosol"]) for p in cparts]
        bb = np.asarray(T.aerosol_button_base())
        from scipy.spatial.transform import Rotation
        bpos = np.asarray(pos) + Rotation.from_quat(quat).apply(bb)
        button = b.add_link(xform=wp.transform(tuple(bpos), wp.quat(*quat)), label=f"{name}_button")
        bparts = T.aerosol_button()
        bshapes = [self._part(button, p, f"{name}_button", 0.6) for p in bparts]
        q_ref = -A["preload"]/A["k"]               # spring rest beyond the top stop: preload at q = 0
        # MuJoCo's joint limits are acceleration-level soft constraints: on a ~3 g button they are ~3 N/mm springs,
        # so the 10 N preload holds it 0.5-0.65 mm above its top stop and 80 N pushes it 0.5-3 mm through the bottom
        # one. A 0.1 kg armature along the slide (inertia of the relative motion only) makes both
        # stops ~30x stiffer with the hard-stop solref of world.py; damping ~critical for it (2 sqrt(k m) = 55 N s/m)
        js = b.add_joint_prismatic(can, button, parent_xform=wp.transform(tuple(bb), wp.quat_identity()),
                                   child_xform=wp.transform((0., 0., 0.), wp.quat_identity()), axis=(0., 0., -1.),
                                   target_pos=q_ref, target_ke=A["k"], target_kd=55.0, limit_lower=0.0,
                                   limit_upper=A["stroke"], armature=0.1, label=f"{name}_button_joint")
        b.add_articulation([jf, js], label=f"{name}_articulation")
        self.bodies[name], self.bodies[f"{name}_button"] = can, button
        self.parts[name], self.parts[f"{name}_button"] = cparts, bparts
        self.items[name] = dict(kind="aerosol", body=can, shapes=[s for s in shapes if s is not None], glides=[])
        self.items[f"{name}_button"] = dict(kind="aerosol_button", body=button, shapes=[s for s in bshapes if s is not None],
                                            glides=[])
        self.joints_extra[f"{name}_button"] = js

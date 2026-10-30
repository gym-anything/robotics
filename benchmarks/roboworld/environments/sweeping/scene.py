"""Physics scene of room 112 (Newton 1.5 ModelBuilder): the installed room, its furniture, the tools and containers
as free rigid bodies, and the catalog robot.

Reuses classroom_arrangement.scene.Scene (the qualified furniture, robot mount, fingertip pads, cameras and
gravity-compensation set-up) and replaces what is room-specific: the installed parts, the furniture list, and the
tools. The lobby dustpan is two bodies (pan, handle) joined by a hinge with stops.
"""

import math
from dataclasses import replace
import numpy as np
from classroom_arrangement.scene import Scene, BUILDERS as FURNITURE, volume
from classroom_arrangement import params as CP
from . import params as P
from . import layout as L
from . import toolkit as T


def quat_mul(a, b):
    """Hamilton product of two quaternions given as (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz)


class SweepScene(Scene):
    def __init__(self, config, newton, wp, spec):
        self.tools = {}                 # name -> dict(kind, body, shapes, parts, spec, [handle_body, hinge])
        self.static_shapes = []
        self.part_of_shape = {}         # shape index -> (owner name, Part)
        self.debris_only = set()        # shapes only the loose-piece solver sees (no rigid contact)
        self.rigid_only = set()         # shapes only the rigid world sees (not colliders of the loose-piece solver)
        self.pad_core_shapes = {}       # hard body of a fingertip under its rubber -> the pad shape (pad_cores)
        super().__init__(config, newton, wp, spec)
        self.tooling()
        self.pad_cores()

    # ------------------------------------------------------------------ room
    def room(self):
        b, wp = self.b, self.wp
        b.add_ground_plane(cfg=self.cfg(mu=P.MU_FLOOR), label="floor")
        self.floor_shape = b.shape_count - 1
        self.materials[b.shape_count - 1] = "floor"
        for name, c, h, mat, mu in L.STATIC:
            i = b.add_shape_box(-1, xform=wp.transform(c, wp.quat_identity()), hx=h[0], hy=h[1], hz=h[2],
                                cfg=self.cfg(mu=mu, density=0.), label=name)
            self.materials[i] = mat
            self.static_shapes.append(i)
        from classroom_arrangement.furniture import _seg
        for name, a, e, r, mat, mu in L.STATIC_CAPSULES:
            c, q, hh = _seg(a, e)
            i = b.add_shape_capsule(-1, xform=wp.transform(c, wp.quat(*q)), radius=r, half_height=hh,
                                    cfg=self.cfg(mu=mu, density=0.), label=name)
            self.materials[i] = mat
            self.static_shapes.append(i)

    # ------------------------------------------------------------------ robot
    def robot(self):
        """The classroom scene's robot set-up with this environment's catalog entry (its default robot and supported
        configurations are those of `sweeping` in robots/catalog.py)."""
        import robots.catalog as catalog
        original = catalog.for_environment
        catalog.for_environment = lambda env, selector: original("sweeping", selector)
        try:
            super().robot()
        finally:
            catalog.for_environment = original

    def pad_cores(self):
        """The hard body of each fingertip under its rubber. The classroom's fingertip pad is one compliant box, the
        Franka Hand finger's tip box with the pad's 5 ms contact, and nothing in it is hard: pressed on harder, it went
        on sinking into what it pressed. A hand brought down on the bare floor at the arm's joint speed pressed one
        pad's corner 2.02 to 2.29 mm into the floor, deeper than its rubber is thick
        (params.PAD_RUBBER), with up to 360 N through it, because MuJoCo sizes that contact on the finger's 85 g while
        about 1 to 4 kg of arm stands behind it. Here the pad keeps its rubber contact as it was, and a box
        PAD_RUBBER smaller on every side, on the same finger, stands for the finger's hard body: it touches nothing
        until the rubber in front of it is pressed flat, and then its contact is rigid (world.py: priority 1, impedance
        PAD_CORE_SOLIMP). It has no mass, no drawing, the pad's friction and the pad's collision filters, and the loose
        pieces do not see it (they meet the pad, world._build_sweeper). It is added after every other shape, so no
        other shape's index changes."""
        b, wp = self.b, self.wp
        filters = list(b.shape_collision_filter_pairs)
        for s in list(self.pad_shapes):
            tf = np.asarray(b.shape_transform[s], float)
            half = np.asarray(b.shape_scale[s], float) - P.PAD_RUBBER
            if (half <= 0).any():
                raise RuntimeError(f"fingertip pad {b.shape_label[s]} is thinner than its rubber ({P.PAD_RUBBER} m)")
            i = b.add_shape_box(int(b.shape_body[s]), xform=wp.transform(tuple(tf[:3]), wp.quat(*tf[3:])), hx=float(half[0]),
                                hy=float(half[1]), hz=float(half[2]),
                                cfg=self.cfg(mu=float(b.shape_material_mu[s]), density=0., gap=float(b.shape_gap[s]), is_visible=False,
                                             collision_group=int(b.shape_collision_group[s])),
                                label=f"{b.shape_label[s]}_core")
            self.materials[i] = self.materials.get(s, "robot")
            self.robot_shapes.append(i)
            self.pad_core_shapes[i] = s
            have = {frozenset(f) for f in b.shape_collision_filter_pairs}       # Newton adds the parent body's own
            for p, q in filters:
                o = q if p == s else p if q == s else None
                if o is not None and frozenset((i, o)) not in have:
                    b.add_shape_collision_filter_pair(i, o)

    # ------------------------------------------------------------------ furniture
    def furniture(self):
        wp, sp = self.wp, self.spec
        for name, it in sp["items"].items():
            kind = it["kind"]
            parts = FURNITURE[kind](sp["glides"])
            x, y, yaw = it["pose"]
            body = self.b.add_body(xform=wp.transform((x, y, 0.0005), wp.quat(*L._yaw_quat(yaw))), label=name)
            self._own(name, kind, body, parts)
        sup = L.supplies()
        for name, it in sp.get("supplies", {}).items():
            x, y, yaw = it["pose"]
            body = self.b.add_body(xform=wp.transform((x, y, it["z"] + 0.0005), wp.quat(*L._yaw_quat(yaw))), label=name)
            self._own(name, "supply", body, sup[it["kind"]])

    def add_part(self, body, p, prefix):
        """As the classroom's, except for the round bottoms of the barrel and the pail and for the caps the brooms
        hang by: their contacts also resist rolling and spinning (MuJoCo condim 6). A disc on a plane touches at one point of its rim once it is tilted,
        and with sliding friction alone that point rolls for ever: a knocked pail stood tilted by 0.6 degree and went
        round at 47 degrees a second for as long as it was watched. A real rim contact resists
        rolling; the two lengths are fractions of the bottom's radius, set from a prior (params)."""
        wp = self.wp
        if p.role == "cap" and p.kind == "capsule" and p.collide:
            # The cap a broom hangs by, on its two prongs: the same fault in another place. Knocked, the broom swung
            # at 1.23 degrees for as long as it was watched (the same amplitude from 4 s to 24 s after the knock): below
            # that the cap rolls on the prongs without sliding, and rolling without slipping loses nothing. With a
            # rolling length of half a millimetre it is at rest 20 s after a knock that swings it 12 degrees.
            cfg = self.cfg(mu=p.mu if p.mu is not None else 0.3, density=p.mass/volume(p) if body >= 0 else 0.,
                           mu_torsional=P.CAP_ROLLING, mu_rolling=P.CAP_ROLLING)
            i = self.b.add_shape_capsule(body, xform=wp.transform(p.pos, wp.quat(*p.quat)), radius=p.size[0], half_height=p.size[1], cfg=cfg,
                                         label=f"{prefix}/{p.name}", custom_attributes={"mujoco:condim": 6})
            self.materials[i] = p.material
            return i
        if p.role == "glide" and p.kind == "cylinder" and p.collide:
            # A glide is an upright cylinder on the floor plane too (classroom_arrangement/furniture.py), and the fault
            # described below is MuJoCo's, not the barrel's. Same cure, in this environment's scene only: the rigid
            # world stands the glide on a square with its corners on the glide's rim (registered as the glide, so that
            # the classroom's friction rule for glides acts on it), and loose pieces go on meeting the round glide.
            half = p.size[0]/math.sqrt(2.0)
            i = super().add_part(body, replace(p, kind="box", size=(half, half, p.size[1])), prefix)
            j = self.b.add_shape_cylinder(body, xform=wp.transform(p.pos, wp.quat(*p.quat)), radius=p.size[0], half_height=p.size[1],
                                          cfg=self.cfg(mu=p.mu if p.mu is not None else 0.3, density=0., has_shape_collision=False),
                                          label=f"{prefix}/{p.name}_round")
            self.materials[j] = p.material
            self.part_of_shape[j] = (prefix, p)
            self.debris_only.add(j)
            self.rigid_only.add(i)
            return i
        if p.role != "container_bottom" or p.kind != "cylinder" or not p.collide:
            return super().add_part(body, p, prefix)
        # THE BOTTOM OF A ROUND CONTAINER DOES NOT STAND ON THE FLOOR AS A CYLINDER. MuJoCo's contact of a cylinder with a
        # plane (mjraw_PlaneCylinder, MuJoCo 3.11) finds the lowest point of the rim from the vector axis (axis . n) - n,
        # normalised. For a disc standing within about 1e-15 to 1e-11 rad of upright that vector is rounding error, its
        # direction is arbitrary and has a part along the plane's normal, and the "lowest rim point" comes out up to
        # millimetres above or below the plane: in 3 to 5 % of such orientations the routine returns no contact at all
        # (measured in isolation; 4 mm of spread in the depths it does return). A disc at rest on a
        # plane settles into exactly that range. The consequence: the barrel of an
        # untouched `broken-mug` start lost its three floor contacts at tick 78515, fell 4.1 mm into the floor in
        # free fall, was caught by its staves and hopped 4.8 mm; the same in `heavy-spill` at 52.3 s.
        # So: the disc is kept as what loose pieces lie on (the debris solver has its own cylinder routine) and for
        # its mass, and takes no part in rigid contact. Four squares of the disc's thickness, turned 22.5 degrees from
        # one another, with their corners on the disc's rim, are what the rigid world stands the container on: sixteen
        # corners, MuJoCo's plane-box contact, which is exact. Their torsional and rolling lengths are the disc's.
        dens = p.mass/volume(p) if body >= 0 else 0.
        i = self.b.add_shape_cylinder(body, xform=wp.transform(p.pos, wp.quat(*p.quat)), radius=p.size[0], half_height=p.size[1],
                                      cfg=self.cfg(mu=p.mu if p.mu is not None else 0.3, density=dens, has_shape_collision=False),
                                      label=f"{prefix}/{p.name}")
        self.materials[i] = p.material
        self.debris_only.add(i)
        half = p.size[0]/math.sqrt(2.0)
        for k in range(4):
            a = k*math.pi/8
            qz = (0., 0., math.sin(a/2), math.cos(a/2))
            q = quat_mul(p.quat, qz)
            cfg = self.cfg(mu=p.mu if p.mu is not None else 0.3, density=0.,
                           mu_torsional=P.BOTTOM_TORSIONAL*p.size[0], mu_rolling=P.BOTTOM_ROLLING*p.size[0])
            j = self.b.add_shape_box(body, xform=wp.transform(p.pos, wp.quat(*q)), hx=half, hy=half, hz=p.size[1], cfg=cfg,
                                     label=f"{prefix}/{p.name}_foot_{k}", custom_attributes={"mujoco:condim": 6})
            self.materials[j] = p.material
            self.rigid_only.add(j)
        return i

    def _own(self, name, kind, body, parts, prefix=None):
        shapes = [self.add_part(body, p, prefix or name) for p in parts]
        for s, p in zip(shapes, parts):
            if s is not None:
                self.part_of_shape[s] = (name, p)
        self.bodies[name] = body
        self.parts[name] = parts
        self.items[name] = dict(kind=kind, body=body, shapes=[s for s in shapes if s is not None],
                                glides=[s for s, p in zip(shapes, parts) if p.role == "glide"])
        return shapes

    # ------------------------------------------------------------------ tools and containers
    def tooling(self):
        wp, b = self.wp, self.b
        for name, t in self.spec["tools"].items():
            spec = T.BUILDERS[t["kind"]](name)
            tf = wp.transform(t["pos"], wp.quat(*t["quat"]))
            if spec.get("handle_parts"):
                # two links: the pan on a free joint, the handle on a hinge across the pan's back
                from scipy.spatial.transform import Rotation
                Rw = Rotation.from_quat(t["quat"])
                ang = float(t.get("hinge", 0.0))
                piv = np.asarray(t["pos"], float) + Rw.apply(spec["pivot"])
                qh = (Rw*Rotation.from_rotvec(np.asarray(spec["pivot_axis"])*ang)).as_quat()
                pan = b.add_link(xform=tf, label=name)
                handle = b.add_link(xform=wp.transform(tuple(piv), wp.quat(*qh)), label=f"{name}_handle")
                j0 = b.add_joint_free(pan, label=f"{name}_free")
                j1 = b.add_joint_revolute(pan, handle, axis=tuple(spec["pivot_axis"]),
                                          parent_xform=wp.transform(spec["pivot"], wp.quat_identity()),
                                          child_xform=wp.transform((0., 0., 0.), wp.quat_identity()),
                                          limit_lower=spec["pivot_limits"][0], limit_upper=spec["pivot_limits"][1],
                                          label=f"{name}_hinge")
                joints, wheel_bodies, wheel_shapes = [j0, j1], [], []
                for k, wh in enumerate(spec.get("wheels", ())):
                    # a wheel: its own link on a hinge along the pan's width (it turns freely; the armature is the
                    # wheel's own moment of inertia once more, for the solver's conditioning)
                    wb = b.add_link(xform=wp.transform(tuple(np.asarray(t["pos"], float) + Rw.apply(wh["pos"])), wp.quat(*t["quat"])),
                                    label=f"{name}_wheel_{k}")
                    jw = b.add_joint_revolute(pan, wb, axis=tuple(wh["axis"]), parent_xform=wp.transform(wh["pos"], wp.quat_identity()),
                                              child_xform=wp.transform((0., 0., 0.), wp.quat_identity()),
                                              armature=0.5*wh["parts"][0].mass*wh["parts"][0].size[0]**2, label=f"{name}_axle_{k}")
                    joints.append(jw)
                    wheel_bodies.append(wb)
                b.add_articulation(joints, label=name)
                b.joint_q[b.joint_q_start[j1]] = ang
                shapes = self._own(name, "tool", pan, spec["parts"])
                hs = self._own(f"{name}_handle", "tool", handle, spec["handle_parts"])
                for k, (wb, wh) in enumerate(zip(wheel_bodies, spec.get("wheels", ()))):
                    wheel_shapes += [s for s in self._own(f"{name}_wheel_{k}", "tool", wb, wh["parts"]) if s is not None]
                self.tools[name] = dict(kind=t["kind"], body=pan, handle_body=handle, hinge=j1, spec=spec,
                                        shapes=[s for s in shapes if s is not None], handle_shapes=[s for s in hs if s is not None],
                                        wheel_bodies=wheel_bodies, wheel_shapes=wheel_shapes)
                # the yoke turns inside the pan's back wall, the wheels beside its side walls: the links of one pan
                # do not collide with each other
                groups = [self.tools[name]["shapes"], self.tools[name]["handle_shapes"], wheel_shapes]
                for a in range(3):
                    for c in range(a + 1, 3):
                        for s0 in groups[a]:
                            for s1 in groups[c]:
                                b.add_shape_collision_filter_pair(s0, s1)
            else:
                body = b.add_body(xform=tf, label=name)
                shapes = self._own(name, "tool", body, spec["parts"])
                self.tools[name] = dict(kind=t["kind"], body=body, spec=spec, shapes=[s for s in shapes if s is not None])
            self.tools[name]["bank"] = spec.get("bank")
